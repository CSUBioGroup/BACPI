import pickle
import sys
import timeit
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn import metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score, precision_recall_curve
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim, 2)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):

        fingerprints, adjacency, words = inputs

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)

        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        T, Y, S = [], [], []
        for data in dataset:
            (correct_labels, predicted_labels,
             predicted_scores) = self.model(data, train=False)
            T.append(correct_labels)
            Y.append(predicted_labels)
            S.append(predicted_scores)
        T, S, Y = np.array(T), np.array(S), np.array(Y)
        auc_value, acc_value, aupr_value = classification_scores(T, S, Y)
        return auc_value, acc_value, aupr_value

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


# def split_dataset(dataset, ratio):
#     n = int(ratio * len(dataset))
#     dataset_1, dataset_2 = dataset[:n], dataset[n:]
#     return dataset_1, dataset_2

def split_dataset(index):
    compounds_ = [compounds[i] for i in index]
    adjacencies_ = [adjacencies[i] for i in index]
    proteins_ = [proteins[i] for i in index]
    interactions_ = [interactions[i] for i in index]
    dataset_ = list(zip(compounds_, adjacencies_, proteins_, interactions_))
    return dataset_


def classification_scores(label, pred_score, pred_label):
    label = label.reshape(-1)
    pred_score = pred_score.reshape(-1)
    pred_label = pred_label.reshape(-1)
    auc = roc_auc_score(label, pred_score)
    acc = accuracy_score(label, pred_label)
    precision, recall, _ = precision_recall_curve(label, pred_label)
    aupr = metrics.auc(recall, precision)
    return round(auc, 6), round(acc, 6), round(aupr, 6)


def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


if __name__ == "__main__":
    radius = '2'
    ngram = '3'
    dim = 10
    layer_gnn = 3
    side = 5
    window = 2 * side + 1
    layer_cnn = 3
    layer_output = 3
    lr = 1e-3
    lr_decay = 0.5
    decay_interval = 10
    weight_decay = 1e-6
    iteration = 20
    (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval, iteration) = \
        map(int, [dim, layer_gnn, window, layer_cnn, layer_output, decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    DATASET = 'human'
    for ratio in ['3', '5']:
        print(DATASET, ratio)
        """Load preprocessed data."""
        dir_input = ('../datasets/' + DATASET + '/cpi/' + ratio + '/')
        compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
        adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
        proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
        interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
        fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
        word_dict = load_pickle(dir_input + 'word_dict.pickle')
        n_fingerprint = len(fingerprint_dict)
        n_word = len(word_dict)

        """Create a dataset and split it into train/dev/test."""
        dataset = list(zip(compounds, adjacencies, proteins, interactions))

        seeds = [7771, 8367, 22, 1812, 4659]
        cv_data = pickle.load(open('../datasets/' + DATASET + '/' + ratio + '/' + DATASET + '_cvdata.txt', 'rb'))
        sample_index = np.load(dir_input + '/sample_ids.npy', allow_pickle=True)
        auc_vec, aupr_vec = [], []
        for seed in seeds:
            for fold in cv_data[seed]:
                sample_ids = [(int(p[0]), str(p[1])) for p in sample_index]
                test_index = [i for i in range(len(sample_ids)) if sample_ids[i] in fold]
                train_dev_index = list(set(range(len(sample_ids))) - set(test_index))
                dev_index = random.sample(list(train_dev_index), int(len(train_dev_index) * 0.125))
                train_index = list(set(train_dev_index) - set(dev_index))

                dataset_train = split_dataset(train_index)
                dataset_dev = split_dataset(dev_index)
                dataset_test = split_dataset(test_index)

                """Set a model."""
                torch.manual_seed(1234)
                model = CompoundProteinInteractionPrediction().to(device)
                trainer = Trainer(model)
                tester = Tester(model)

                AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tAUC_test\tacc_test\taupr_test')
                print(AUCs)
                start = timeit.default_timer()
                max_auc = 0.0
                for epoch in range(1, iteration):
                    if epoch % decay_interval == 0:
                        trainer.optimizer.param_groups[0]['lr'] *= lr_decay

                    loss_train = trainer.train(dataset_train)
                    AUC_dev = tester.test(dataset_dev)[0]
                    AUC_test, acc_test, aupr_test = tester.test(dataset_test)

                    if max_auc < AUC_dev:
                        max_auc = AUC_dev
                        res = [AUC_test, acc_test, aupr_test]

                    end = timeit.default_timer()
                    time = end - start

                    AUCs = [epoch, time, loss_train, AUC_dev, AUC_test, acc_test, aupr_test]
                    print('\t'.join(map(str, AUCs)))
                auc_vec.append(res[0])
                aupr_vec.append(res[2])

        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)

        print('Test avg auc:{}, aupr:{}, auc_conf:{}, aupr_conf:{}'.format(auc_avg, aupr_avg, auc_conf, aupr_conf))
        pd.DataFrame(auc_vec).to_csv("../output/cpipre_auc_" + DATASET + ratio + ".txt", header=False, index=False)
        pd.DataFrame(aupr_vec).to_csv("../output/cpipre_aupr_" + DATASET + ratio + ".txt", header=False, index=False)
