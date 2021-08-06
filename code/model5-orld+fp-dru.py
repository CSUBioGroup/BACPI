import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import pickle
import math
import numpy as np
import torch.optim as optim
from utils import params, batch2tensor, regression_scores, init_embed
from sklearn.model_selection import KFold
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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
        self.W_out = nn.ModuleList([nn.Linear(3 * latent_dim, 3 * latent_dim) for _ in range(layer_out)])

        if task == 'interaction':
            self.predict = nn.Linear(2 * latent_dim, 2)
        elif task == 'affinity':
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
        # self.W_out = nn.Linear(comp_dim * comp_dim * 2, 1)

        # fingerprint module
        self.fps_first = nn.Linear(1024, comp_dim)
        self.fps_layer = nn.ModuleList([nn.Linear(comp_dim, comp_dim) for _ in range(3)])

        self.W_pair_att = nn.Parameter(torch.empty(size=(comp_dim, comp_dim)))
        nn.init.xavier_uniform_(self.W_pair_att.data, gain=1.414)

    def comp_gat(self, atoms, atoms_mask, adj):
        atoms_vector = self.embedding_layer_atom(atoms)
        atoms_multi_head = torch.cat([gat(atoms_vector, adj) for gat in self.gat_layers], dim=2)
        # atoms_multi_head = F.dropout(atoms_multi_head, self.dropout, training=self.training)
        atoms_vector = F.elu(self.gat_out(atoms_multi_head, adj))
        atoms_vector = F.leaky_relu(self.W_comp(atoms_vector), self.alpha)
        return atoms_vector

    def prot_cnn(self, amino, amino_mask):
        amino_vector = self.embedding_layer_amino(amino)
        amino_vector = torch.unsqueeze(amino_vector, 1)
        for i in range(self.layer_cnn):
            amino_vector = F.leaky_relu(self.conv_layers[i](amino_vector), self.alpha)
            # amino_vector = F.dropout(amino_vector, self.dropout, training=self.training)
        amino_vector = torch.squeeze(amino_vector, 1)
        amino_vector = F.leaky_relu(self.W_prot(amino_vector), self.alpha)
        return amino_vector

    def bidirectional_attention_prediction(self, atoms_vector, atoms_mask, amino_vector, amino_mask):
        b = atoms_vector.shape[0]

        atoms_vector = F.leaky_relu(self.W_attention(atoms_vector), self.alpha)
        amino_vector = F.leaky_relu(self.W_attention(amino_vector), self.alpha)

        prot_vector = torch.sum(amino_vector * amino_mask.view(b, -1, 1), dim=1) / torch.sum(amino_mask, dim=1, keepdim=True)
        prot_rep = torch.unsqueeze(prot_vector, 1).repeat_interleave(atoms_vector.shape[1], dim=1)
        Wh_atoms2prot = torch.cat([atoms_vector, prot_rep], dim=2)

        atoms_attention = torch.tanh(self.att_atoms2prot(Wh_atoms2prot))
        atoms_vector = atoms_vector * atoms_attention
        comp_vector = torch.sum(atoms_vector * atoms_mask.view(b, -1, 1), dim=1) / torch.sum(atoms_mask, dim=1, keepdim=True)

        comp_rep = torch.unsqueeze(comp_vector, 1).repeat_interleave(amino_vector.shape[1], dim=1)
        Wh_amino2comp = torch.cat([amino_vector, comp_rep], dim=2)

        amino_attention = torch.tanh((self.att_amino2comp(Wh_amino2comp)))
        amino_vector = amino_vector * amino_attention
        prot_vector = torch.sum(amino_vector * amino_mask.view(b, -1, 1), dim=1) / torch.sum(amino_mask, dim=1, keepdim=True)

        comp_prot_vector = torch.cat((comp_vector, prot_vector), dim=1)
        for i in range(self.layer_out):
            comp_prot_vector = F.leaky_relu(self.W_out[i](comp_prot_vector), self.alpha)
        return self.predict(comp_prot_vector)

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

        c0 = torch.sum(comp_feats * vertex_mask, dim=1) / torch.sum(vertex_mask, dim=1)
        p0 = torch.sum(prot_feats * seq_mask, dim=1) / torch.sum(seq_mask, dim=1)

        m = c0 * p0
        for DMA_iter in range(self.DMA_depth):
            c_to_p = torch.matmul(pairwise_pred.transpose(1, 2),
                                  F.tanh(self.c_to_p_transform[DMA_iter](comp_feats)))  # batch * n_residue * hidden
            p_to_c = torch.matmul(pairwise_pred,
                                  F.tanh(self.p_to_c_transform[DMA_iter](prot_feats)))  # batch * n_vertex * hidden

            c_tmp = F.tanh(self.hc0[DMA_iter](comp_feats)) * F.tanh(self.mc1[DMA_iter](m)).view(batch_size, 1,
                                                                                                -1) * p_to_c
            p_tmp = F.tanh(self.hp0[DMA_iter](prot_feats)) * F.tanh(self.mp1[DMA_iter](m)).view(batch_size, 1,
                                                                                                -1) * c_to_p

            c_att = self.mask_softmax(self.hc1[DMA_iter](c_tmp).view(batch_size, -1), vertex_mask.view(batch_size, -1))
            p_att = self.mask_softmax(self.hp1[DMA_iter](p_tmp).view(batch_size, -1), seq_mask.view(batch_size, -1))

            cf = torch.sum(comp_feats * c_att.view(batch_size, -1, 1), dim=1)
            pf = torch.sum(prot_feats * p_att.view(batch_size, -1, 1), dim=1)

            m = self.GRU_dma(m, cf * pf)

        return cf, pf

    def forward(self, atoms, atoms_mask, adjacency, amino, amino_mask, fps):
        batch_size = atoms.shape[0]
        # atoms_vector = self.comp_gat(atoms, atoms_mask, adjacency)
        # amino_vector = self.prot_cnn(amino, amino_mask)
        # # prediction = self.bidirectional_attention_prediction(atoms_vector, atoms_mask, amino_vector, amino_mask)
        # super_feature = torch.sum(atoms_vector * atoms_mask.view(batch_size, -1, 1), dim=1, keepdim=True)
        # pairwise_pred = self.Pairwise_pred_module(batch_size, atoms_vector, amino_vector, atoms_mask, amino_mask)
        # prediction = self.Affinity_pred_module(batch_size, atoms_vector, amino_vector, super_feature, atoms_mask,
        #                                        amino_mask, pairwise_pred)

        atoms_vector = self.comp_gat(atoms, atoms_mask, adjacency)
        comp_vector1 = torch.sum(atoms_vector * atoms_mask.view(batch_size, -1, 1), dim=1) / torch.sum(atoms_mask, dim=1, keepdim=True)

        comp_vector = F.leaky_relu(self.fps_first(fps), 0.1)
        for i in range(3):
            comp_vector = F.leaky_relu(self.fps_layer[i](comp_vector), 0.1)
            # comp_vector = F.dropout(comp_vector, self.dropout, training=self.training)
        amino_vector = self.prot_cnn(amino, amino_mask)
        tmp = torch.matmul(amino_vector, self.W_pair_att)
        e = F.leaky_relu(torch.bmm(tmp, torch.unsqueeze(comp_vector, 2)), self.alpha)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(torch.unsqueeze(amino_mask, 2) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        prot_vector = torch.bmm(attention.transpose(1, 2), amino_vector)
        prot_vector = torch.squeeze(prot_vector, 1)
        comp_prot_vector = torch.cat((comp_vector, comp_vector1, prot_vector), dim=1)
        for i in range(self.layer_out):
            comp_prot_vector = F.leaky_relu(self.W_out[i](comp_prot_vector), self.alpha)
        prediction = self.predict(comp_prot_vector)
        return prediction


def print2file(buf, outFile, p=False):
    if p:
        print(buf)
    outfd = open(outFile, 'a+')
    outfd.write(buf + '\n')
    outfd.close()


def fps2tensor(arr):
    new_arr = np.zeros((arr.shape[0], 1024))
    for i, a in enumerate(arr):
        new_arr[i, :] = np.array(list(a), dtype=int)
    return Variable(torch.FloatTensor(new_arr)).to(device)


def train_eval(model, task, train_data, valid_data, test_data, device, params, train_fps, valid_fps, test_fps):
    criterion = F.mse_loss if task == 'affinity' else F.cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    idx = np.arange(len(train_data[0]))
    batch_size = params.batch_size
    # min_loss = 1000
    for epoch in range(params.num_epochs):
        print2file('epoch:{}'.format(epoch), 'bi-model.txt', True)
        np.random.shuffle(idx)
        model.train()
        for i in range(math.ceil(len(train_data[0]) / batch_size)):
            batch_data = [train_data[di][idx[i * batch_size: (i + 1) * batch_size]] for di in range(4)]
            atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, label = batch2tensor(batch_data, device)
            batch_fps = fps2tensor(train_fps[idx[i * batch_size: (i + 1) * batch_size]])
            pred = model(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps)
            loss = criterion(pred.float(), label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write('\repoch:{}, batch:{}/{}, loss:{}'
                             .format(epoch, i, math.ceil(len(train_data[0])/batch_size)-1, float(loss.data)))
            sys.stdout.flush()

        rmse_train, pearson_train, spearman_train = test(model, task, train_data, batch_size, device, train_fps)
        info = '\nTrain rmse:{}, pearson:{}, spearman:{}'.format(rmse_train, pearson_train, spearman_train)
        print2file(info, 'bi-model.txt', True)

        rmse_valid, pearson_valid, spearman_valid = test(model, task, valid_data, batch_size, device, valid_fps)
        info = 'Valid rmse:{}, pearson:{}, spearman:{}'.format(rmse_valid, pearson_valid, spearman_valid)
        print2file(info, 'bi-model.txt', True)

        rmse_test, pearson_test, spearman_test = test(model, task, test_data, batch_size, device, test_fps)
        info = 'Test rmse:{}, pearson:{}, spearman:{}'.format(rmse_test, pearson_test, spearman_test)
        print2file(info, 'bi-model.txt', True)

        scheduler.step()


def test(model, task, test_data, batch_size, device, test_fps):
    model.eval()
    predictions = []
    labels = []
    for i in range(math.ceil(len(test_data[0]) / batch_size)):
        batch_data = [test_data[di][i * batch_size: (i + 1) * batch_size] for di in range(4)]
        atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, label = batch2tensor(batch_data, device)
        batch_fps = fps2tensor(test_fps[i * batch_size: (i + 1) * batch_size])
        with torch.no_grad():
            pred = model(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps)
        if task == 'affinity':
            predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += label.cpu().numpy().reshape(-1).tolist()
        else:
            pass
    predictions = np.array(predictions)
    labels = np.array(labels)
    if task == 'affinity':
        rmse_value, pearson_value, spearman_value = regression_scores(labels, predictions)
        return round(rmse_value, 6), round(pearson_value, 6), round(spearman_value, 6)
    else:
        pass


if __name__ == '__main__':

    # params.lr = 0.001
    # params.step_size = 10
    task = 'affinity'
    DATASET = 'BindingDB'
    measure = 'IC50'  # 'Ki', 'IC50', 'Kd', 'EC50'
    target_class = 'test'  # 'GPCR', 'ER', 'channel', 'kinase'

    dir_input = '../../CPI_prediction/dataset/BindingDB/input/radius2_ngram3/'

    device = torch.device('cuda') if params.mode == 'gpu' and torch.cuda.is_available() else torch.device('cpu')
    print('The code run on the', device)

    print('Load data...')
    compounds = np.load(dir_input + 'compounds.npy', allow_pickle=True)
    adjacencies = np.load(dir_input + 'adjacencies.npy', allow_pickle=True)
    fingerprint = np.load('../datasets/BindingDB/IC50_test/fingerprint.npy', allow_pickle=True)
    proteins = np.load(dir_input + 'proteins.npy', allow_pickle=True)
    interactions = np.load(dir_input + 'interactions.npy', allow_pickle=True)
    dataset = [compounds, adjacencies, proteins, interactions]

    atom_dict = pickle.load(open(dir_input + 'fingerprint_dict.pickle', 'rb'))
    amino_dict = pickle.load(open(dir_input + 'word_dict.pickle', 'rb'))
    n_atom = len(atom_dict)
    n_amino = len(amino_dict)

    print('training...')
    kf = KFold(5, shuffle=True)
    for train_valid_idx, test_idx in kf.split(range(len(dataset[0]))):
        model = BiDACPI(task, n_atom, n_amino, params)
        model.to(device)

        valid_idx = np.random.choice(train_valid_idx, int(len(train_valid_idx) * 0.125), replace=False)
        train_idx = list(set(train_valid_idx) - set(valid_idx))
        print('train num:', len(train_idx), 'valid num:', len(valid_idx), 'test num:', len(test_idx))

        train_data = [dataset[i][train_idx] for i in range(4)]
        valid_data = [dataset[i][valid_idx] for i in range(4)]
        test_data = [dataset[i][test_idx] for i in range(4)]

        train_fps = fingerprint[train_idx]
        valid_fps = fingerprint[valid_idx]
        test_fps = fingerprint[test_idx]

        train_eval(model, task, train_data, valid_data, test_data, device, params, train_fps, valid_fps, test_fps)
