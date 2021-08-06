from collections import defaultdict
import os
import pickle
import sys

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))

    atoms_set = set(range(mol.GetNumAtoms()))
    isolate_atoms = atoms_set - set(i_jbond_dict.keys())
    bond = bond_dict['nan']
    for a in isolate_atoms:
        i_jbond_dict[a].append((a, bond))

    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])

            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    adjacency = np.array(adjacency)
    adjacency += np.eye(adjacency.shape[0], dtype=int)
    return adjacency


def get_fp(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=True)
    return fp.ToBitString()


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)


def get_gos(go):
    return np.array(list(map(int, go.split(','))))


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


def precess_affinity(target_class):
    print("load original data...")
    raw_data = pd.read_csv('../data/' + measure + '/' + target_class, sep='\t', usecols=list(range(49)),
                           dtype=str)
    data = raw_data[['PubChem CID', 'UniProt (SwissProt) Primary ID of Target Chain', 'Ligand SMILES',
                     'BindingDB Target Chain  Sequence',
                     'pKi_[M]', 'pIC50_[M]', 'pKd_[M]', 'pEC50_[M]']]
    data.columns = ['CID', 'UID', 'SMILES', 'Sequence', 'Ki', 'IC50', 'Kd', 'EC50']
    data = data[['CID', 'UID', 'SMILES', 'Sequence', measure]]

    compounds, adjacencies, fps, proteins, interactions = [], [], [], [], []

    print("generate input data...")
    for index in range(len(data)):
        smiles, sequence, interaction = data.iloc[index, 2:]

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
        compounds.append(fingerprints)

        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency)

        fps.append(get_fp(mol))

        words = split_sequence(sequence, ngram)
        proteins.append(words)

        interactions.append(np.array([float(interaction)]))

    print("save input data...")
    dir_input = ('../datasets/' + DATASET + '/' + measure + '/' + target_class + '/')
    os.makedirs(dir_input, exist_ok=True)

    np.save(dir_input + 'compounds', compounds)
    np.save(dir_input + 'adjacencies', adjacencies)
    np.save(dir_input + 'fingerprint', fps)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'interactions', interactions)


def process_interaction(file_name, dir_input):
    with open(file_name, 'r') as f:
        data_list = f.read().strip().split('\n')

    # data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)

    compounds, adjacencies, fps, proteins, gos, interactions = [], [], [], [], [], []

    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))

        smiles, sequence, go, interaction = data.strip().split()

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.
        # mol = Chem.MolFromSmiles(smiles)
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
        compounds.append(fingerprints)

        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency)

        fps.append(get_fp(mol))

        words = split_sequence(sequence, ngram)
        proteins.append(words)

        gos.append(get_gos(go))

        interactions.append(np.array([float(interaction)]))

    print("save input data...")
    os.makedirs(dir_input, exist_ok=True)

    np.save(dir_input + 'compounds', compounds)
    np.save(dir_input + 'adjacencies', adjacencies)
    np.save(dir_input + 'fingerprint', fps)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'gos', gos)
    np.save(dir_input + 'interactions', interactions)


if __name__ == "__main__":
    radius, ngram = 2, 3
    DATASET = 'IBM'
    ratio = '5'
    measure = 'IC50'  # 'Ki', 'IC50', 'Kd', 'EC50'

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))

    # for target_class in ['train', 'test', 'GPCR', 'ER', 'channel', 'kinase']:
    #     precess_affinity(target_class)
    #
    # dump_dictionary(fingerprint_dict, '../datasets/' + DATASET + '/' + measure + '/atom_dict')
    # dump_dictionary(word_dict, '../datasets/' + DATASET + '/' + measure + '/amino_dict')

    # process_interaction('../data/' + DATASET + '/' + ratio + '/data.txt', '../datasets/' + DATASET + '/' + ratio + '/')
    # dump_dictionary(fingerprint_dict, '../datasets/' + DATASET + '/' + ratio + '/atom_dict')
    # dump_dictionary(word_dict, '../datasets/' + DATASET + '/' + ratio + '/amino_dict')

    for name in ['train', 'dev', 'test', 'seenProt', 'unseenProt']:
        filename = '../data/' + DATASET + '/' + name + '/data.txt'
        dirinput = '../datasets/' + DATASET + '/' + name + '/'
        process_interaction(filename, dirinput)

    dump_dictionary(fingerprint_dict, '../datasets/' + DATASET + '/atom_dict')
    dump_dictionary(word_dict, '../datasets/' + DATASET + '/amino_dict')
    print('The preprocess of ' + DATASET + ' dataset has finished!')
