import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import pickle
import math
import numpy as np
import pandas as pd
import torch.optim as optim
from utils import params, batch2tensor, regression_scores, classification_scores, load_data, cv_split, split_data
from utils import mean_confidence_interval
from sklearn.model_selection import KFold
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# params.lr = 0.001
# params.step_size = 10
# params.gamma = 0.5
params.num_epochs = 15
params.comp_dim = 64
params.prot_dim = 64
params.latent_dim = 64
params.gat_dim = 40
params.batch_size = 5
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

        self.fp0 = nn.Parameter(torch.empty(size=(1024, latent_dim)))
        nn.init.xavier_uniform_(self.fp0, gain=1.414)
        self.fp1 = nn.Parameter(torch.empty(size=(latent_dim, latent_dim)))
        nn.init.xavier_uniform_(self.fp1, gain=1.414)

        self.bidat_num = 4

        self.U = nn.ParameterList([nn.Parameter(torch.empty(size=(latent_dim, latent_dim))) for _ in range(self.bidat_num)])
        for i in range(self.bidat_num):
            nn.init.xavier_uniform_(self.U[i], gain=1.414)

        self.transform_c2p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.transform_p2c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])

        self.bihidden_c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.bihidden_p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.biatt_c = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.bidat_num)])
        self.biatt_p = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.bidat_num)])

        self.comb_c = nn.Linear(latent_dim * self.bidat_num, latent_dim)
        self.comb_p = nn.Linear(latent_dim * self.bidat_num, latent_dim)

        if task == 'affinity':
            self.output = nn.Linear(latent_dim * latent_dim * 2, 1)
        elif task == 'interaction':
            self.output = nn.Linear(latent_dim * latent_dim * 2, 2)

    def comp_gat(self, atoms, atoms_mask, adj):
        atoms_vector = self.embedding_layer_atom(atoms)
        atoms_multi_head = torch.cat([gat(atoms_vector, adj) for gat in self.gat_layers], dim=2)
        atoms_vector = F.elu(self.gat_out(atoms_multi_head, adj))
        atoms_vector = F.leaky_relu(self.W_comp(atoms_vector), self.alpha)
        return atoms_vector

    def prot_cnn(self, amino, amino_mask):
        amino_vector = self.embedding_layer_amino(amino)
        amino_vector = torch.unsqueeze(amino_vector, 1)
        for i in range(self.layer_cnn):
            amino_vector = F.leaky_relu(self.conv_layers[i](amino_vector), self.alpha)
        amino_vector = torch.squeeze(amino_vector, 1)
        amino_vector = F.leaky_relu(self.W_prot(amino_vector), self.alpha)
        return amino_vector

    def mask_softmax(self, a, mask, dim=-1):
        a_max = torch.max(a, dim, keepdim=True)[0]
        a_exp = torch.exp(a - a_max)
        a_exp = a_exp * mask
        a_softmax = a_exp / (torch.sum(a_exp, dim, keepdim=True) + 1e-6)
        return a_softmax

    def bidirectional_attention_prediction(self, atoms_vector, atoms_mask, fps, amino_vector, amino_mask):
        b = atoms_vector.shape[0]
        for i in range(self.bidat_num):
            A = torch.tanh(torch.matmul(torch.matmul(atoms_vector, self.U[i]), amino_vector.transpose(1, 2)))
            # A = torch.sigmoid(torch.matmul(atoms_vector, amino_vector.transpose(1, 2)))
            A = A * torch.matmul(atoms_mask.view(b, -1, 1), amino_mask.view(b, 1, -1))

            atoms_trans = torch.matmul(A, torch.tanh(self.transform_p2c[i](amino_vector)))
            amino_trans = torch.matmul(A.transpose(1, 2), torch.tanh(self.transform_c2p[i](atoms_vector)))

            atoms_tmp = torch.cat([torch.tanh(self.bihidden_c[i](atoms_vector)), atoms_trans], dim=2)
            amino_tmp = torch.cat([torch.tanh(self.bihidden_p[i](amino_vector)), amino_trans], dim=2)

            atoms_att = self.mask_softmax(self.biatt_c[i](atoms_tmp).view(b, -1), atoms_mask.view(b, -1))
            amino_att = self.mask_softmax(self.biatt_p[i](amino_tmp).view(b, -1), amino_mask.view(b, -1))

            cf = torch.sum(atoms_vector * atoms_att.view(b, -1, 1), dim=1)
            pf = torch.sum(amino_vector * amino_att.view(b, -1, 1), dim=1)

            if i == 0:
                cat_cf = cf
                cat_pf = pf
            else:
                cat_cf = torch.cat([cat_cf.view(b, -1), cf.view(b, -1)], dim=1)
                cat_pf = torch.cat([cat_pf.view(b, -1), pf.view(b, -1)], dim=1)

        cf_final = torch.cat([self.comb_c(cat_cf).view(b, -1), fps.view(b, -1)], dim=1)
        pf_final = self.comb_p(cat_pf)
        cf_pf = F.leaky_relu(torch.matmul(cf_final.view(b, -1, 1), pf_final.view(b, 1, -1)).view(b, -1), 0.1)
        return self.output(cf_pf)

    def forward(self, atoms, atoms_mask, adjacency, amino, amino_mask, fps):
        batch_size = atoms.shape[0]
        atoms_vector = self.comp_gat(atoms, atoms_mask, adjacency)
        amino_vector = self.prot_cnn(amino, amino_mask)

        super_feature = F.leaky_relu(torch.matmul(fps, self.fp0), 0.1)
        super_feature = F.leaky_relu(torch.matmul(super_feature, self.fp1), 0.1)

        prediction = self.bidirectional_attention_prediction(atoms_vector, atoms_mask, super_feature, amino_vector, amino_mask)
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
    max_auc = 0.0
    for epoch in range(params.num_epochs):
        # print2file('epoch:{}'.format(epoch), prefix + 'BACPI_IC50_log.txt', True)
        print(epoch)
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
            auc_train, acc_train, apur_train = classification_scores(labels, predictions, pred_labels)
            info = 'Train auc:{}, acc:{}, aupr:{}'.format(auc_train, acc_train, apur_train)
            print(info)

            auc_dev, acc_dev, aupr_dev = test(model, task, data_dev, batch_size, device)
            info = 'Dev auc:{}, acc:{}, aupr:{}'.format(auc_dev, acc_dev, aupr_dev)
            print(info)

            auc_test, acc_test, aupr_test = test(model, task, data_test, batch_size, device)
            info = 'Test auc:{}, acc:{}, aupr:{}'.format(auc_test, acc_test, aupr_test)
            print(info)

            if max_auc < auc_dev:
                max_auc = auc_dev
                res = [auc_test, acc_test, aupr_test]

        scheduler.step()
    return res


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
        auc_value, acc_value, aupr_value = classification_scores(labels, predictions, pred_labels)
        return auc_value, acc_value, aupr_value


def main(data_name, ratio):
    task = 'interaction'
    device = torch.device('cuda') if params.mode == 'gpu' and torch.cuda.is_available() else torch.device('cpu')
    print('The code run on the', device)

    print('Load data...')
    data_dir = '../datasets/' + data_name + '/' + ratio
    data = load_data(data_dir, None)
    atom_dict = pickle.load(open(data_dir + '/atom_dict', 'rb'))
    amino_dict = pickle.load(open(data_dir + '/amino_dict', 'rb'))

    seeds = [7771, 8367, 22, 1812, 4659]
    cv_data = pickle.load(open(data_dir + '/' + data_name + '_cvdata.txt', 'rb'))
    sample_ids = np.load(data_dir + '/sample_ids.npy', allow_pickle=True)
    auc_vec, aupr_vec = [], []
    for seed in seeds:
        for fold in cv_data[seed]:
            data_train_dev, data_test = cv_split(fold, data, sample_ids)
            # data_train, data_test = cv_split(fold, data, sample_ids)
            data_train, data_dev = split_data(data_train_dev)
            model = BiDACPI(task, len(atom_dict), len(amino_dict), params)
            model.to(device)
            res = train_eval(model, task, data_train, data_dev, data_test, device, params)
            # res = train_eval(model, task, data_train, None, data_test, device, params)
            auc_vec.append(res[0])
            aupr_vec.append(res[2])

    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)

    print('Test avg auc:{}, aupr:{}, auc_conf:{}, aupr_conf:{}'.format(auc_avg, aupr_avg, auc_conf, aupr_conf))
    pd.DataFrame(auc_vec).to_csv("../output/BiDAT_auc_" + data_name + ratio + ".txt", header=False, index=False)
    pd.DataFrame(aupr_vec).to_csv("../output/BiDAT_aupr_" + data_name + ratio + ".txt", header=False, index=False)


if __name__ == '__main__':
    # main(data_name='human', ratio='1')
    dataset = 'celegans'
    for ratio in ['5']:
        print(dataset, ratio)
        main(dataset, ratio)
