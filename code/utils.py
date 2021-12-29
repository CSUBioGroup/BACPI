import numpy as np
from math import sqrt
from scipy import stats
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
import argparse
import torch
from torch.autograd import Variable

args = argparse.ArgumentParser(description='Argparse for compound-protein interactions prediction')
args.add_argument('-mode', default='gpu', help='gpu/cpu')
args.add_argument('-cuda', default='0', help='visible cuda devices')
args.add_argument('-lr', type=float, default=0.001, help='init learning rate')
args.add_argument('-step_size', type=int, default=15, help='step size of lr_scheduler')
args.add_argument('-gamma', type=float, default=0.5, help='lr weight decay rate')
args.add_argument('-batch_size', type=int, default=32, help='batch size')
args.add_argument('-num_epochs', type=int, default=25, help='number of epochs')

# graph attention layer
args.add_argument('-gat_dim', type=int, default=50, help='dimension of node feature in graph attention layer')
args.add_argument('-num_head', type=int, default=3, help='number of graph attention layer head')
args.add_argument('-dropout', type=float, default=0.1)
args.add_argument('-alpha', type=float, default=0.1, help='LeakyReLU alpha')

args.add_argument('-comp_dim', type=int, default=80, help='dimension of compound atoms feature')
args.add_argument('-prot_dim', type=int, default=80, help='dimension of protein amino feature')
args.add_argument('-latent_dim', type=int, default=80, help='dimension of compound and protein feature')

args.add_argument('-window', type=int, default=5, help='window size of cnn model')
args.add_argument('-layer_cnn', type=int, default=4, help='number of layer in cnn model')
args.add_argument('-layer_out', type=int, default=4, help='number of output layer in prediction model')

params, _ = args.parse_known_args()


def batch_pad(arr):
    N = max([a.shape[0] for a in arr])
    if arr[0].ndim == 1:
        new_arr = np.zeros((len(arr), N))
        new_arr_mask = np.zeros((len(arr), N))
        for i, a in enumerate(arr):
            n = a.shape[0]
            new_arr[i, :n] = a + 1
            new_arr_mask[i, :n] = 1
        return new_arr, new_arr_mask

    elif arr[0].ndim == 2:
        new_arr = np.zeros((len(arr), N, N))
        new_arr_mask = np.zeros((len(arr), N, N))
        for i, a in enumerate(arr):
            n = a.shape[0]
            new_arr[i, :n, :n] = a
            new_arr_mask[i, :n, :n] = 1
        return new_arr, new_arr_mask


def fps2number(arr):
    new_arr = np.zeros((arr.shape[0], 1024))
    for i, a in enumerate(arr):
        new_arr[i, :] = np.array(list(a), dtype=int)
    return new_arr


def batch2tensor(batch_data, device):
    atoms_pad, atoms_mask = batch_pad(batch_data[0])
    adjacencies_pad, _ = batch_pad(batch_data[1])
    fps = fps2number(batch_data[2])
    amino_pad, amino_mask = batch_pad(batch_data[3])

    atoms_pad = Variable(torch.LongTensor(atoms_pad)).to(device)
    atoms_mask = Variable(torch.FloatTensor(atoms_mask)).to(device)
    adjacencies_pad = Variable(torch.LongTensor(adjacencies_pad)).to(device)
    fps = Variable(torch.FloatTensor(fps)).to(device)
    amino_pad = Variable(torch.LongTensor(amino_pad)).to(device)
    amino_mask = Variable(torch.FloatTensor(amino_mask)).to(device)

    label = torch.FloatTensor(batch_data[4]).to(device)

    return atoms_pad, atoms_mask, adjacencies_pad, fps, amino_pad, amino_mask, label


def load_data(datadir, target_type):
    if target_type:
        dir_input = datadir + '/' + target_type + '/'
    else:
        dir_input = datadir + '/'
    compounds = np.load(dir_input + 'compounds.npy', allow_pickle=True)
    adjacencies = np.load(dir_input + 'adjacencies.npy', allow_pickle=True)
    fingerprint = np.load(dir_input + 'fingerprint.npy', allow_pickle=True)
    proteins = np.load(dir_input + 'proteins.npy', allow_pickle=True)
    interactions = np.load(dir_input + 'interactions.npy', allow_pickle=True)
    data_pack = [compounds, adjacencies, fingerprint, proteins, interactions]
    return data_pack


def split_data(train_data, ratio=0.1):
    idx = np.arange(len(train_data[0]))
    np.random.shuffle(idx)
    num_dev = int(len(train_data[0]) * ratio)
    idx_dev, idx_train = idx[:num_dev], idx[num_dev:]
    data_train = [train_data[di][idx_train] for di in range(len(train_data))]
    data_dev = [train_data[di][idx_dev] for di in range(len(train_data))]
    return data_train, data_dev


def regression_scores(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    rmse = sqrt(((label - pred)**2).mean(axis=0))
    pearson = np.corrcoef(label, pred)[0, 1]
    spearman = stats.spearmanr(label, pred)[0]
    return round(rmse, 6), round(pearson, 6), round(spearman, 6)


def classification_scores(label, pred_score, pred_label):
    label = label.reshape(-1)
    pred_score = pred_score.reshape(-1)
    pred_label = pred_label.reshape(-1)
    auc = roc_auc_score(label, pred_score)
    acc = accuracy_score(label, pred_label)
    precision, recall, _ = precision_recall_curve(label, pred_label)
    aupr = metrics.auc(recall, precision)
    return round(auc, 6), round(acc, 6), round(aupr, 6)
