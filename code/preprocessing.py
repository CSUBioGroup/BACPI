import os
import tarfile
import pandas as pd


def extract_data(measure, file):
    data = pd.read_csv(file, sep='\t', usecols=list(range(49)), dtype=str)
    data = data[['Ligand SMILES', 'BindingDB Target Chain  Sequence', 'pKi_[M]', 'pIC50_[M]', 'pKd_[M]', 'pEC50_[M]']]
    data.columns = ['SMILES', 'Sequence', 'Ki', 'IC50', 'Kd', 'EC50']
    data = data[['SMILES', 'Sequence', measure]]
    data.to_csv(file + '.txt', index=None, header=None)


def preprocess_affinity(dataset):
    data_dir = '../data/affinity/' + dataset
    if not os.path.isdir(data_dir):
        tar = tarfile.open(data_dir + '.tar.xz')
        os.mkdir(data_dir)
        for name in tar.getnames():
            tar.extract(name, '../data/affinity/')
        tar.close()

    extract_data(dataset, data_dir + '/train')
    extract_data(dataset, data_dir + '/test')


def preprocess_interaction(dataset):
    pass


def preprocess(task, dataset):
    if task == 'affinity':
        preprocess_affinity(dataset)
    else:
        preprocess_interaction(dataset)


if __name__ == '__main__':
    for dataset in ['IC50', 'EC50', 'Ki', 'Kd']:
        preprocess_affinity(dataset)

    for dataset in ['human', 'celegans']:
        preprocess_interaction(dataset)
