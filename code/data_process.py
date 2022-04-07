import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
from data_prepare import training_data_prepare


atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
word_dict = defaultdict(lambda: len(word_dict))


def create_atoms(mol):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
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


def atom_features(atoms, i_jbond_dict, radius):
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]
    
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])

            nodes = fingerprints
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


def get_fingerprints(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=True)
    return fp.ToBitString()


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


def extract_input_data(input_path, output_path, radius, ngram):
    data = pd.read_csv(input_path + '.txt', header=None)
    compounds, adjacencies, fps, proteins, interactions = [], [], [], [], []

    for index in range(len(data)):
        smiles, sequence, interaction = data.iloc[index, :]

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        compounds.append(atom_features(atoms, i_jbond_dict, radius))
        adjacencies.append(create_adjacency(mol))
        fps.append(get_fingerprints(mol))
        proteins.append(split_sequence(sequence, ngram))
        interactions.append(np.array([float(interaction)]))

    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, 'compounds'), compounds)
    np.save(os.path.join(output_path, 'adjacencies'), adjacencies)
    np.save(os.path.join(output_path, 'fingerprint'), fps)
    np.save(os.path.join(output_path, 'proteins'), proteins)
    np.save(os.path.join(output_path, 'interactions'), interactions)


def training_data_process(task, dataset):
    radius, ngram = 2, 3

    if not os.path.isdir(os.path.join('../data', task, dataset)):
        training_data_prepare(task, dataset)

    for name in ['train', 'test']:
        input_path = os.path.join('../data', task, dataset, name)
        output_path = os.path.join('../datasets', task, dataset, name)
        extract_input_data(input_path, output_path, radius, ngram)

    dump_dictionary(fingerprint_dict, os.path.join('../datasets', task, dataset, 'atom_dict'))
    dump_dictionary(word_dict, os.path.join('../datasets', task, dataset, 'amino_dict'))
