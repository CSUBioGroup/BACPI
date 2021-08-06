import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import pickle
import math
import numpy as np
import torch.optim as optim
from utils import params, batch2tensor, regression_scores, classification_scores, load_data, split_data, init_embed
from sklearn.model_selection import KFold
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# params.lr = 0.001
# params.step_size = 10
# params.gamma = 0.5
params.comp_dim = 64
params.prot_dim = 64
params.latent_dim = 64
params.gat_dim = 40
params.batch_size = 10
params.layer_cnn = 3

class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(3), self.alpha)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, Wh)

        return F.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        b = Wh.size()[0]
        N = Wh.size()[1]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat_interleave(N, dim=0).view(b, N*N, self.out_features)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)

        return all_combinations_matrix.view(b, N, N, 2 * self.out_features)


class BiDACPI(nn.Module):
    def __init__(self, task, n_atom, n_amino, params):
        super(BiDACPI, self).__init__()

        comp_dim, prot_dim, gat_dim, num_head, dropout, alpha, window, layer_cnn, latent_dim, layer_out = \
            params.comp_dim, params.prot_dim, params.gat_dim, params.num_head, params.dropout, params.alpha,\
            params.window, params.layer_cnn, params.latent_dim, params.layer_out

        self.embedding_layer_atom = nn.Embedding(n_atom+1, comp_dim)
        self.embedding_layer_amino = nn.Embedding(n_amino+1, prot_dim)

        self.dropout = dropout
        self.alpha = alpha
        self.layer_cnn = layer_cnn
        self.layer_out = layer_out

        self.gat_layers = [GATLayer(comp_dim, gat_dim, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(num_head)]
        for i, layer in enumerate(self.gat_layers):
            self.add_module('gat_layer_{}'.format(i), layer)
        self.gat_out = GATLayer(gat_dim * num_head, comp_dim, dropout=dropout, alpha=alpha, concat=False)
        self.W_comp = nn.Linear(comp_dim, latent_dim)

        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2*window+1,
                                                    stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_prot = nn.Linear(prot_dim, latent_dim)

        self.W_attention = nn.Linear(latent_dim, latent_dim)
        self.att_atoms2prot = nn.Linear(2 * latent_dim, 1)
        self.att_amino2comp = nn.Linear(2 * latent_dim, 1)
        # self.fully_connect = nn.ModuleList([nn.Linear(2 * latent_dim, 2 * latent_dim) for _ in range(layer_out)])
        self.fully_connect = nn.ModuleList([nn.Linear(3 * latent_dim, 3 * latent_dim) for _ in range(layer_out)])

        if task == 'interaction':
            self.predict = nn.Linear(2 * latent_dim, 2)
        elif task == 'affinity':
            # self.predict = nn.Linear(2 * latent_dim, 1)
            self.predict = nn.Linear(3 * latent_dim, 1)

        self.pairwise_compound = nn.Linear(comp_dim, comp_dim)
        self.pairwise_protein = nn.Linear(prot_dim, prot_dim)

        self.super_final = nn.Linear(comp_dim, comp_dim)
        self.c_final = nn.Linear(comp_dim, comp_dim)
        self.p_final = nn.Linear(prot_dim, prot_dim)

        self.DMA_depth = 4

        self.c_to_p_transform = nn.ModuleList([nn.Linear(comp_dim, comp_dim) for i in range(self.DMA_depth)])
        self.p_to_c_transform = nn.ModuleList([nn.Linear(prot_dim, prot_dim) for i in range(self.DMA_depth)])

        self.mc0 = nn.Linear(comp_dim, comp_dim)
        self.mp0 = nn.Linear(comp_dim, comp_dim)

        self.mc1 = nn.ModuleList([nn.Linear(comp_dim, comp_dim) for i in range(self.DMA_depth)])
        self.mp1 = nn.ModuleList([nn.Linear(comp_dim, comp_dim) for i in range(self.DMA_depth)])

        self.hc0 = nn.ModuleList([nn.Linear(comp_dim, comp_dim) for i in range(self.DMA_depth)])
        self.hp0 = nn.ModuleList([nn.Linear(comp_dim, comp_dim) for i in range(self.DMA_depth)])
        self.hc1 = nn.ModuleList([nn.Linear(comp_dim, 1) for i in range(self.DMA_depth)])
        self.hp1 = nn.ModuleList([nn.Linear(comp_dim, 1) for i in range(self.DMA_depth)])

        self.GRU_dma = nn.GRUCell(comp_dim, comp_dim)
        # Output layer
        if task == 'affinity':
            self.W_out = nn.Linear(comp_dim * comp_dim * 2, 1)
        elif task == 'interaction':
            self.W_out = nn.Linear(comp_dim * comp_dim * 2, 2)

        # fingerprint module
        self.fps_first = nn.Linear(1024, comp_dim)
        self.fps_layer = nn.ModuleList([nn.Linear(comp_dim, comp_dim) for _ in range(3)])

        self.W_pair_att = nn.Parameter(torch.empty(size=(comp_dim, comp_dim)))
        nn.init.xavier_uniform_(self.W_pair_att.data, gain=1.414)

        self.U = nn.Parameter(torch.empty(size=(comp_dim, prot_dim)))
        nn.init.xavier_uniform_(self.U.data, gain=1.414)

        self.fp0 = nn.Parameter(torch.empty(size=(1024, comp_dim)))
        nn.init.xavier_uniform_(self.fp0, gain=1.414)

        self.fp1 = nn.Parameter(torch.empty(size=(comp_dim, comp_dim)))
        nn.init.xavier_uniform_(self.fp1, gain=1.414)

    def comp_gat(self, atoms, atoms_mask, adj):
        atoms_vector = self.embedding_layer_atom(atoms)
        atoms_multi_head = torch.cat([gat(atoms_vector, adj) for gat in self.gat_layers], dim=2)
        # atoms_multi_head = F.dropout(atoms_multi_head, self.dropout, training=self.training)
        atoms_vector = F.elu(self.gat_out(atoms_multi_head, adj))
        atoms_vector = F.leaky_relu(self.W_comp(atoms_vector), self.alpha)
        return atoms_vector * atoms_mask.view(atoms_mask.shape[0], -1, 1)
        # return atoms_vector

    def prot_cnn(self, amino, amino_mask):
        amino_vector = self.embedding_layer_amino(amino)
        amino_vector = torch.unsqueeze(amino_vector, 1)
        for i in range(self.layer_cnn):
            amino_vector = F.leaky_relu(self.conv_layers[i](amino_vector), self.alpha)
            # amino_vector = F.dropout(amino_vector, self.dropout, training=self.training)
        amino_vector = torch.squeeze(amino_vector, 1)
        amino_vector = F.leaky_relu(self.W_prot(amino_vector), self.alpha)
        return amino_vector * amino_mask.view(amino_mask.shape[0], -1, 1)
        # return amino_vector

    def mask_softmax(self, a, mask, dim=-1):
        a_max = torch.max(a, dim, keepdim=True)[0]
        a_exp = torch.exp(a - a_max)
        a_exp = a_exp * mask
        a_softmax = a_exp / (torch.sum(a_exp, dim, keepdim=True) + 1e-6)
        return a_softmax

    def Pairwise_pred_module(self, batch_size, comp_feature, prot_feature, vertex_mask, seq_mask):

        pairwise_c_feature = F.leaky_relu(self.pairwise_compound(comp_feature), 0.1)
        pairwise_p_feature = F.leaky_relu(self.pairwise_protein(prot_feature), 0.1)
        pairwise_pred = torch.sigmoid(torch.matmul(pairwise_c_feature, pairwise_p_feature.transpose(1, 2)))
        # pairwise_pred = torch.tanh(
        #     torch.matmul(torch.matmul(pairwise_c_feature, self.U), pairwise_p_feature.transpose(1, 2)))
        pairwise_mask = torch.matmul(vertex_mask.view(batch_size, -1, 1), seq_mask.view(batch_size, 1, -1))
        pairwise_pred = pairwise_pred * pairwise_mask

        return pairwise_pred

    def Affinity_pred_module(self, batch_size, comp_feature, prot_feature, super_feature, vertex_mask, seq_mask,
                             pairwise_pred):

        comp_feature = F.leaky_relu(self.c_final(comp_feature), 0.1)
        prot_feature = F.leaky_relu(self.p_final(prot_feature), 0.1)
        super_feature = F.leaky_relu(self.super_final(super_feature.view(batch_size, -1)), 0.1)

        cf, pf = self.dma_gru(batch_size, comp_feature, vertex_mask, prot_feature, seq_mask, pairwise_pred)

        cf = torch.cat([cf.view(batch_size, -1), super_feature.view(batch_size, -1)], dim=1)
        kroneck = F.leaky_relu(
            torch.matmul(cf.view(batch_size, -1, 1), pf.view(batch_size, 1, -1)).view(batch_size, -1), 0.1)

        affinity_pred = self.W_out(kroneck)
        return affinity_pred

    def dma_gru(self, batch_size, comp_feats, vertex_mask, prot_feats, seq_mask, pairwise_pred):
        vertex_mask = vertex_mask.view(batch_size, -1, 1)
        seq_mask = seq_mask.view(batch_size, -1, 1)

        cf = torch.sum(comp_feats * vertex_mask, dim=1) / torch.sum(vertex_mask, dim=1)
        pf = torch.sum(prot_feats * seq_mask, dim=1) / torch.sum(seq_mask, dim=1)

        m = cf * pf
        for DMA_iter in range(self.DMA_depth):
            c_to_p = torch.matmul(pairwise_pred.transpose(1, 2),
                                  torch.tanh(self.c_to_p_transform[DMA_iter](comp_feats)))  # batch * n_residue * hidden
            p_to_c = torch.matmul(pairwise_pred,
                                  torch.tanh(self.p_to_c_transform[DMA_iter](prot_feats)))  # batch * n_vertex * hidden

            c_tmp = torch.tanh(self.hc0[DMA_iter](comp_feats)) * torch.tanh(self.mc1[DMA_iter](m)).view(batch_size, 1,
                                                                                                -1) * p_to_c
            p_tmp = torch.tanh(self.hp0[DMA_iter](prot_feats)) * torch.tanh(self.mp1[DMA_iter](m)).view(batch_size, 1,
                                                                                                -1) * c_to_p

            c_att = self.mask_softmax(self.hc1[DMA_iter](c_tmp).view(batch_size, -1), vertex_mask.view(batch_size, -1))
            p_att = self.mask_softmax(self.hp1[DMA_iter](p_tmp).view(batch_size, -1), seq_mask.view(batch_size, -1))

            cf = torch.sum(comp_feats * c_att.view(batch_size, -1, 1), dim=1)
            pf = torch.sum(prot_feats * p_att.view(batch_size, -1, 1), dim=1)

            m = self.GRU_dma(m, cf * pf)

        return cf, pf

    def bidirectional_attention_prediction(self, atoms_vector, atoms_mask, fps, amino_vector, amino_mask):
        # 1 fp -> prot att -> comp att -> gru

        pass

    def forward(self, atoms, atoms_mask, adjacency, amino, amino_mask, fps):
        batch_size = atoms.shape[0]
        atoms_vector = self.comp_gat(atoms, atoms_mask, adjacency)
        amino_vector = self.prot_cnn(amino, amino_mask)

        # super_feature = torch.sum(atoms_vector * atoms_mask.view(batch_size, -1, 1), dim=1, keepdim=True)
        super_feature = F.leaky_relu(torch.matmul(fps, self.fp0), 0.1)
        super_feature = F.leaky_relu(torch.matmul(super_feature, self.fp1), 0.1)

        pairwise_pred = self.Pairwise_pred_module(batch_size, atoms_vector, amino_vector, atoms_mask, amino_mask)
        prediction = self.Affinity_pred_module(batch_size, atoms_vector, amino_vector, super_feature, atoms_mask, amino_mask, pairwise_pred)

        # tmp = torch.matmul(amino_vector, self.W_pair_att)
        # e = F.leaky_relu(torch.bmm(tmp, torch.unsqueeze(comp_vector, 2)), self.alpha)
        # zero_vec = -9e15 * torch.ones_like(e)
        # attention = torch.where(torch.unsqueeze(amino_mask, 2) > 0, e, zero_vec)
        # attention = F.softmax(attention, dim=1)
        # # attention = F.dropout(attention, self.dropout, training=self.training)
        # prot_vector = torch.bmm(attention.transpose(1, 2), amino_vector)
        # prot_vector = torch.squeeze(prot_vector, 1)
        # comp_prot_vector = torch.cat((comp_vector, prot_vector), dim=1)
        # for i in range(self.layer_out):
        #     comp_prot_vector = F.leaky_relu(self.fully_connect[i](comp_prot_vector), self.alpha)
        # prediction = self.predict(comp_prot_vector)
        return prediction


def print2file(buf, outFile, p=False):
    if p:
        print(buf)
    outfd = open(outFile, 'a+')
    outfd.write(buf + '\n')
    outfd.close()


def train_eval(model, task, data_train, data_dev, data_test, device, params):
    criterion = F.mse_loss if task == 'affinity' else F.cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    prefix = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    # print2file('gat+cnn+monn+fp', prefix + 'BACPI_IC50_log.txt', True)
    idx = np.arange(len(data_train[0]))
    batch_size = params.batch_size
    min_loss = 1000
    for epoch in range(params.num_epochs):
        # print2file('epoch:{}'.format(epoch), prefix + 'BACPI_IC50_log.txt', True)
        np.random.shuffle(idx)
        model.train()
        pred_labels = []
        predictions = []
        labels = []
        for i in range(math.ceil(len(data_train[0]) / batch_size)):
            batch_data = [data_train[di][idx[i * batch_size: (i + 1) * batch_size]] for di in range(len(data_train))]
            atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad, amino_mask, label = batch2tensor(batch_data, device)
            pred = model(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps)
            if task == 'affinity':
                loss = criterion(pred.float(), label.float())
                predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
                labels += label.cpu().numpy().reshape(-1).tolist()
            elif task == 'interaction':
                loss = criterion(pred.float(), label.view(label.shape[0]).long())
                ys = F.softmax(pred, 1).to('cpu').data.numpy()
                pred_labels += list(map(lambda x: np.argmax(x), ys))
                predictions += list(map(lambda x: x[1], ys))
                labels += label.cpu().numpy().reshape(-1).tolist()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write('\repoch:{}, batch:{}/{}, loss:{}'.format(
                epoch, i, math.ceil(len(data_train[0])/batch_size)-1, float(loss.data)))
            sys.stdout.flush()

        if task == 'affinity':
            print(' ')
            predictions = np.array(predictions)
            labels = np.array(labels)
            rmse_train, pearson_train, spearman_train = regression_scores(labels, predictions)
            info = 'Train rmse:{}, pearson:{}, spearman:{}'.format(rmse_train, pearson_train, spearman_train)
            print2file(info, prefix + 'BACPI_IC50_log.txt', True)

            rmse_dev, pearson_dev, spearman_dev = test(model, task, data_dev, batch_size, device)
            info = 'Dev rmse:{}, pearson:{}, spearman:{}'.format(rmse_dev, pearson_dev, spearman_dev)
            print2file(info, prefix + 'BACPI_IC50_log.txt', True)

            rmse_test, pearson_test, spearman_test = test(model, task, data_test, batch_size, device)
            info = 'Test rmse:{}, pearson:{}, spearman:{}'.format(rmse_test, pearson_test, spearman_test)
            print2file(info, prefix + 'BACPI_IC50_log.txt', True)

            if rmse_dev < min_loss:
                min_loss = rmse_dev
                torch.save(model, '../checkpoint/best_model.pth')

        else:
            print(' ')
            pred_labels = np.array(pred_labels)
            predictions = np.array(predictions)
            labels = np.array(labels)
            auc_train, recall_train, precision_train = classification_scores(labels, predictions, pred_labels)
            info = 'Train auc:{}, recall:{}, precision:{}'.format(auc_train, recall_train, precision_train)
            print(info)

            # auc_dev, recall_dev, precision_dev = test(model, task, data_dev, batch_size, device)
            # info = 'Dev auc:{}, recall:{}, precision:{}'.format(auc_dev, recall_dev, precision_dev)
            # print(info)

            auc_test, recall_test, precision_test = test(model, task, data_test, batch_size, device)
            info = 'Test auc:{}, recall:{}, precision:{}'.format(auc_test, recall_test, precision_test)
            print(info)

        scheduler.step()


def test(model, task, data_test, batch_size, device):
    model.eval()
    predictions = []
    pred_labels = []
    labels = []
    for i in range(math.ceil(len(data_test[0]) / batch_size)):
        batch_data = [data_test[di][i * batch_size: (i + 1) * batch_size] for di in range(len(data_test))]
        atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad, amino_mask, label = batch2tensor(batch_data, device)
        with torch.no_grad():
            pred = model(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps)
        if task == 'affinity':
            predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += label.cpu().numpy().reshape(-1).tolist()
        else:
            ys = F.softmax(pred, 1).to('cpu').data.numpy()
            pred_labels += list(map(lambda x: np.argmax(x), ys))
            predictions += list(map(lambda x: x[1], ys))
            labels += label.cpu().numpy().reshape(-1).tolist()
    pred_labels = np.array(pred_labels)
    predictions = np.array(predictions)
    labels = np.array(labels)
    if task == 'affinity':
        rmse_value, pearson_value, spearman_value = regression_scores(labels, predictions)
        return rmse_value, pearson_value, spearman_value
    else:
        auc_value, recall_value, precision_value = classification_scores(labels, predictions, pred_labels)
        return auc_value, recall_value, precision_value


if __name__ == '__main__':

    task = 'interaction'
    DATASET = 'celegans'
    data_dir = '../datasets/' + DATASET

    device = torch.device('cuda') if params.mode == 'gpu' and torch.cuda.is_available() else torch.device('cpu')
    print('The code run on the', device)

    print('Load data...')
    data = load_data(data_dir, None)
    # data_train, data_ = split_data(data, 0.2)
    # test_data, data_dev = split_data(data_, 0.5)
    data_train, test_data = split_data(data, 0.1)

    atom_dict = pickle.load(open(data_dir + '/atom_dict', 'rb'))
    amino_dict = pickle.load(open(data_dir + '/amino_dict', 'rb'))

    print('training...')
    model = BiDACPI(task, len(atom_dict), len(amino_dict), params)
    model.to(device)
    # train_eval(model, task, data_train, data_dev, test_data, device, params)
    train_eval(model, task, data_train, None, test_data, device, params)

    print('Finish training!')
    # best_model = torch.load('../checkpoint/best_model.pth')
