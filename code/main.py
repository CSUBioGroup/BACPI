import sys
import pickle
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import params, batch2tensor, regression_scores, init_embed
from model import BiDACPI
from sklearn.model_selection import KFold
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def print2file(buf, outFile, p=False):
    if p:
        print(buf)
    outfd = open(outFile, 'a+')
    outfd.write(buf + '\n')
    outfd.close()


def train_eval(model, task, train_data, valid_data, test_data, device, params):
    criterion = F.mse_loss if task == 'affinity' else F.cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    idx = np.arange(len(train_data[0]))
    batch_size = params.batch_size
    # min_loss = 1000
    for epoch in range(params.num_epochs):
        print2file('epoch:{}'.format(epoch), 'bi-model.txt', True)
        np.random.shuffle(idx)
        for i in range(math.ceil(len(train_data[0]) / batch_size)):
            batch_data = [train_data[di][idx[i * batch_size: (i + 1) * batch_size]] for di in range(4)]
            atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, label = batch2tensor(batch_data, device)
            pred = model(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask)
            loss = criterion(pred.float(), label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write('\repoch:{}, batch:{}/{}, loss:{}'
                             .format(epoch, i, math.ceil(len(train_data[0])/batch_size)-1, float(loss.data)))
            sys.stdout.flush()

        rmse_train, pearson_train, spearman_train = test(model, task, train_data, batch_size, device)
        info = '\nTrain rmse:{}, pearson:{}, spearman:{}'.format(rmse_train, pearson_train, spearman_train)
        print2file(info, 'bi-model.txt', True)

        rmse_valid, pearson_valid, spearman_valid = test(model, task, valid_data, batch_size, device)
        info = 'Valid rmse:{}, pearson:{}, spearman:{}'.format(rmse_valid, pearson_valid, spearman_valid)
        print2file(info, 'bi-model.txt', True)

        rmse_test, pearson_test, spearman_test = test(model, task, test_data, batch_size, device)
        info = 'Test rmse:{}, pearson:{}, spearman:{}'.format(rmse_test, pearson_test, spearman_test)
        print2file(info, 'bi-model.txt', True)

        scheduler.step()


def test(model, task, test_data, batch_size, device):
    model.eval()
    predictions = []
    labels = []
    for i in range(math.ceil(len(test_data[0]) / batch_size)):
        batch_data = [test_data[di][i * batch_size: (i + 1) * batch_size] for di in range(4)]
        atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, label = batch2tensor(batch_data, device)
        with torch.no_grad():
            pred = model(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask)
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
    task = 'affinity'
    DATASET = 'BindingDB'
    measure = 'IC50'  # 'Ki', 'IC50', 'Kd', 'EC50'
    target_class = 'test'  # 'GPCR', 'ER', 'channel', 'kinase'

    dir_input = ('../datasets/' + DATASET + '/' + measure + '_' + target_class + '/')

    device = torch.device('cuda') if params.mode == 'gpu' and torch.cuda.is_available() else torch.device('cpu')
    print('The code run on the', device)

    print('Load data...')
    compounds = np.load(dir_input + 'compounds.npy', allow_pickle=True)
    adjacencies = np.load(dir_input + 'adjacencies.npy', allow_pickle=True)
    fingerprint = np.load(dir_input + 'fingerprint.npy', allow_pickle=True)
    proteins = np.load(dir_input + 'proteins.npy', allow_pickle=True)
    interactions = np.load(dir_input + 'interactions.npy', allow_pickle=True)
    dataset = [compounds, adjacencies, proteins, interactions]

    init_atom, init_amino = init_embed(dir_input)

    print('training...')
    kf = KFold(5, shuffle=True)
    for train_valid_idx, test_idx in kf.split(range(len(dataset[0]))):
        model = BiDACPI(task, init_atom, init_amino, params)
        model.to(device)

        valid_idx = np.random.choice(train_valid_idx, int(len(train_valid_idx) * 0.125), replace=False)
        train_idx = list(set(train_valid_idx) - set(valid_idx))
        print('train num:', len(train_idx), 'valid num:', len(valid_idx), 'test num:', len(test_idx))

        train_data = [dataset[i][train_idx] for i in range(4)]
        valid_data = [dataset[i][valid_idx] for i in range(4)]
        test_data = [dataset[i][test_idx] for i in range(4)]

        train_eval(model, task, train_data, valid_data, test_data, device, params)
