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
from utils import params, batch2tensor, regression_scores, classification_scores, load_data, split_data, init_embed
from case_study import BiDACPI
from case_study import GATLayer
from sklearn.model_selection import KFold
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def test(model, task, data_test, batch_size, device):
    model.eval()
    predictions = []
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
            pass
    predictions = np.array(predictions)
    labels = np.array(labels)
    if task == 'affinity':
        rmse_value, pearson_value, spearman_value = regression_scores(labels, predictions)
        return rmse_value, pearson_value, spearman_value, predictions
    else:
        pass


if __name__ == '__main__':
    device = torch.device('cuda') if params.mode == 'gpu' and torch.cuda.is_available() else torch.device('cpu')
    task = 'affinity'
    data_dir = '../datasets/case_study'
    CIDs = pd.read_csv('../datasets/case_study/CIDs.csv')
    cids = list(CIDs['PubChem CID'])
    pre_model = torch.load('../checkpoint/ic50_case_study2.pt')
    for case in ['O60885', 'Q9H8M2', 'Q06187', 'P50613', 'P00533', 'Q15910', 'P22455', 'P16234', 'P07949', 'P10721']:
        case_data = load_data(data_dir, case)
        _, _, _, pred_res = test(pre_model, task, case_data, params.batch_size, device)
        pred_res = list(pred_res)
        cid_pred = list(zip(cids, pred_res))
        cid_pred = sorted(cid_pred, key=lambda item: item[1], reverse=True)
        cid_pred_pd = pd.DataFrame(cid_pred, index=None)
        cid_pred_pd.to_csv('../results/case_study/cids_' + case + '_2.csv', index=False, header=['CID', 'affinity'])
