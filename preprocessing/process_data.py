from collections import defaultdict
import os
import pickle
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from scipy.cluster.hierarchy import fcluster, linkage, single
from scipy.spatial.distance import pdist

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
             'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
             'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U',
             'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'unknown']
aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list)
                    + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                    + onek_encoding_unk(atom.GetExplicitValence(), [1, 2, 3, 4, 5, 6])
                    + onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
                    + [atom.GetIsAromatic()], dtype=np.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array(
        [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
         bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)


def Mol2Graph(mol):
    idxfunc = lambda x: x.GetIdx()
    n_atoms = mol.GetNumAtoms()
    assert mol.GetNumBonds() >= 0

    fatoms = np.zeros((n_atoms,), dtype=np.int32)

    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        fatoms[idx] = atom_dict[''.join(str(x) for x in atom_features(atom).astype(int).tolist())]

    adjacency = Chem.GetAdjacencyMatrix(mol) + np.eye(n_atoms, dtype=int)
    return fatoms, adjacency


def Protein2Sequence(sequence, ngram=1):
    sequence = sequence.upper()
    word_list = [sequence[i:i + ngram] for i in range(len(sequence) - ngram + 1)]
    output = []
    for word in word_list:
        if word not in aa_list:
            output.append(amino_dict['X'])
        else:
            output.append(amino_dict[word])
    if ngram == 3:
        output = [-1] + output + [-1]  # pad
    return np.array(output, np.int32)


def get_fp(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=True)
    return fp.ToBitString()


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


if __name__ == "__main__":
    DATASET = 'BindingDB'
    measure = 'IC50'  # 'Ki', 'IC50', 'Kd', 'EC50'
    target_class = 'test'  # 'GPCR', 'ER', 'channel', 'kinase'

    print("load original data...")
    raw_data = pd.read_csv('../data/' + measure + '/' + target_class, sep='\t', usecols=list(range(49)),
                           dtype=str)
    data = raw_data[['PubChem CID', 'UniProt (SwissProt) Primary ID of Target Chain', 'Ligand SMILES',
                     'BindingDB Target Chain  Sequence',
                     'pKi_[M]', 'pIC50_[M]', 'pKd_[M]', 'pEC50_[M]']]
    data.columns = ['CID', 'UID', 'SMILES', 'Sequence', 'Ki', 'IC50', 'Kd', 'EC50']
    data = data[['CID', 'UID', 'SMILES', 'Sequence', measure]]

    atom_dict = defaultdict(lambda: len(atom_dict))
    amino_dict = defaultdict(lambda: len(amino_dict))
    for aa in aa_list:
        amino_dict[aa]
    amino_dict['X']

    compounds, adjacencies, fps, proteins, interactions = [], [], [], [], []

    print("generate input data...")
    for index in range(len(data)):
        smiles, sequence, interaction = data.iloc[index, 2:]

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms, adj = Mol2Graph(mol)
        fp = get_fp(mol)
        compounds.append(atoms)
        adjacencies.append(adj)
        fps.append(fp)

        amino = Protein2Sequence(sequence, ngram=1)
        proteins.append(amino)

        interactions.append(np.array([float(interaction)]))

    print("save input data...")
    dir_input = ('../datasets/' + DATASET + '/' + measure + '_' + target_class + '/')
    os.makedirs(dir_input, exist_ok=True)

    np.save(dir_input + 'compounds', compounds)
    np.save(dir_input + 'adjacencies', adjacencies)
    np.save(dir_input + 'fingerprint', fps)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'interactions', interactions)
    dump_dictionary(atom_dict, dir_input + 'atom_dict')
    dump_dictionary(amino_dict, dir_input + 'amino_dict')

    print('The preprocess of ' + DATASET + ' dataset has finished!')
