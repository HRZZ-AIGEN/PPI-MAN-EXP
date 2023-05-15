from abc import ABC
from logging import root
import pandas as pd
import numpy as np
import json
import networkx as nx
import torch
import os

from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import BondStereo as BS
from rdkit.Chem.rdchem import BondDir as BD
from pathlib import Path
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset, Batch
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import pairwise_distances
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
from torch_geometric.typing import (
    EdgeType,
    NodeType,
    OptTensor
)
from typing import (
    Any,
    Dict,
    List
)


class MolData(Data):

    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None, 
        edge_attr: OptTensor = None, y: OptTensor = None, pos: OptTensor = None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, item, *args, **kwargs):
        if key == 'dist_matrix' or key == 'adj_matrix' or key == 'name':
            return None
        else:
            return super().__cat_dim__(key, item, *args, **kwargs)


class MolecularGraphs(InMemoryDataset, ABC):
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    stereo = {BS.STEREONONE: 0, BS.STEREOANY: 1, BS.STEREOZ: 2,
              BS.STEREOE: 3, BS.STEREOCIS: 4, BS.STEREOTRANS: 5}
    direction = {BD.NONE: 0, BD.BEGINWEDGE: 1, BD.BEGINDASH: 2,
                 BD.ENDDOWNRIGHT: 3, BD.ENDUPRIGHT: 4, BD.EITHERDOUBLE: 5,
                 BD.UNKNOWN: 6}

    def __init__(self, root):
        super().__init__(root)
        self.root = Path(root)
        # torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])
        # print(self.data, self.slices)

    @property
    def raw_dir(self):
        return Path(self.root) / 'data/raw/NCI60June2022'

    @property
    def processed_dir(self):
        return Path(self.root) / 'data/processed/pytorch_graphs/NCI60'

    @property
    def raw_file_names(self):
        return "NCI60_IC50_PUBCHEM_PUBLIC.pkl"

    @property
    def processed_file_names(self):
        return "nci60_molecular_graphs_public.pt"
    
    def atoms_features(self, mol, feats):
        """ Features """

        atomic_number, aromatic, donor, acceptor, s  = [], [], [], [], []
        sp, sp2, sp3, sp3d, sp3d2, num_hs = [], [], [], [], [], [] 
        for atom in mol.GetAtoms():
            # type_idx.append(self.types[atom.GetSymbol()])
            atomic_number.append(atom.GetAtomicNum())
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybridization = atom.GetHybridization()
            donor.append(0)
            acceptor.append(0)
            s.append(1 if hybridization == HybridizationType.S
                    else 0)
            sp.append(1 if hybridization == HybridizationType.SP
                    else 0)
            sp2.append(1 if hybridization == HybridizationType.SP2
                    else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3
                    else 0)
            sp3d.append(1 if hybridization == HybridizationType.SP3D
                        else 0)
            sp3d2.append(1 if hybridization == HybridizationType.SP3D2
                        else 0)

            num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))

        for j in range(0, len(feats)):
            if feats[j].GetFamily() == 'Donor':
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    donor[k] = 1
            elif feats[j].GetFamily() == 'Acceptor':
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    acceptor[k] = 1
        # return the features of nodes in molecular dataset
        x = torch.tensor([atomic_number,
                            acceptor,
                            donor,
                            aromatic,
                            s, sp, sp2, sp3, sp3d, sp3d2,
                            num_hs],
                            dtype=torch.float).t().contiguous()
        return x

    def edges_indexes(self, mol):
        row, col, bond_idx, bond_stereo, bond_dir = [], [], [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            # bond_idx += 2 * [self.bonds[bond.GetBondType()]]
            # bond_stereo += 2 * [self.stereo[bond.GetStereo()]]
            # bond_dir += 2 * [self.direction[bond.GetBondDir()]]
            # 2* list, because the bonds are defined 2 times, start -> end,
            # and end -> start
        # return edge_index
        return torch.tensor([row, col], dtype=torch.long)
    

    def other_feature_names(self):
        with open('/workspace/data/raw/desc.json', 'r', encoding='utf-8') as f:
            desc = json.load(f)
        
        return desc['rf_importance_features']


    def dump_to_json(self, data, names):
        with open("tmp_molecules.json", "w") as final:
            json.dump(data, final)
        with open("tmp_names.json", "w") as final:
            json.dump(names, final)
        return
    

    def read_from_json(self):
        names = json.load("tmp/names.json")
        return json.load("tmp/molecules.json"), names, len(names)

    
    def process(self):
        """ Load data """
        # bad_cids = [117560, 23939, 0, 23976]  # don't save these molecules
        labels = pd.read_pickle(self.raw_paths[0])
        labels = labels.groupby('pubchem_id', as_index=False).first()
        # labels = labels.loc[~labels['pubchem_id'].isin(bad_cids)]
        fdef_name = Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef'
        factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))

        molecules = list(labels['pubchem_id'].unique())
        molecule_list = []
        names = []
        data_list = []
        # other_feature_names = self.other_feature_names()
        print("Creating molecular graphs")
        for molecule in enumerate(tqdm((molecules),
                                       total=len(molecules),
                                       position=0,
                                       leave=True)):
            pubchem_id = molecule[1]
            smiles = labels.loc[labels['pubchem_id'] == pubchem_id]['isomeric_smiles']
            # other_features = labels.loc[labels['pubchem_id'] == name][other_feature_names].values[0]
            
            mol = Chem.MolFromSmiles(smiles.values[0])  # there are multiple returned values

            """ Features """
            try:
                feats = factory.GetFeaturesForMol(mol)
                x = self.atoms_features(mol, feats)
                edge_index = self.edges_indexes(mol)
                
                try:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, maxAttempts=1000)
                    AllChem.UFFOptimizeMolecule(mol)
                    mol = Chem.RemoveHs(mol)
                except:
                    AllChem.Compute2DCoords(mol)

                conf = mol.GetConformer()
                pos_matrix = np.array(
                    [[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                    for k in range(mol.GetNumAtoms())])
                dist_matrix = pairwise_distances(pos_matrix)

                """ Create adjecency matrix """
                adj_matrix = np.eye(mol.GetNumAtoms())
                for bond in mol.GetBonds():
                    begin_atom = bond.GetBeginAtom().GetIdx()
                    end_atom = bond.GetEndAtom().GetIdx()
                    adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1
                # Here we can add attirbutes about the molecule, as fingerprint, smiles, ... 
                # Other attributes could enhance drug representation but contribute to uninterpretability
                # 'defined_bond_stereo_count', 'molecular_weight', 'covalent_unit_count', 'bond_stereo_count', 
                # 'heavy_atom_count', 'rotatable_bond_count'
                # needs to be calculated: 'xlogp', 'MlogP'   
                keys = ['x', 'pubchem_id', 'distance_matrix', 'adj_matrix', 'edge_index'] 
                values = [x, pubchem_id, dist_matrix, adj_matrix, edge_index] 
                # keys = ['x', 'name', 'distance_matrix', 'adj_matrix', 'edge_index'] + other_feature_names
                # values = [x, name, dist_matrix, adj_matrix, edge_index] + list(other_features)
                molecule_dict = dict(zip(keys, values))
                molecule_list.append(molecule_dict)
                names.append(pubchem_id)

            except:
                print("name: {}, has a bug".format(names[-1]))

        print("Concatenating the dataset\n\n")
        # self.dump_to_json(molecule_list)     
        for molecule_data in molecule_list:
            pubchem_id = molecule_data.get("pubchem_id")
            x = molecule_data.get("x")
            distance_matrix = molecule_data.get("distance_matrix")
            adj_matrix = molecule_data.get("adj_matrix")
            edge_index = molecule_data.get("edge_index")

            # Omit the targets one molecule has many different targets depending on the cell line 
            data = MolData(
                x=x, 
                edge_index=edge_index,
                adj_matrix=adj_matrix, 
                dist_matrix=distance_matrix, 
                pubchem_id = pubchem_id,
                # If you add other fearures to the dataset enlargeing the data, 
                # can be used for better predictions less interpretable,
                # Domain expertese needed how to encode the data to te graph :
                # all nodes or just chosen ones, rotable bonds feature on bonds? etc...

                # complexity = molecule_data.get('complexity'), 
                # exact_mass=molecule_data.get('exact_mass'),
                # monoisotopic_mass=molecule_data.get('monoisotopic_mass'),
                # tpsa=molecule_data.get('tpsa'),
                # xlogp=molecule_data.get('xlogp'),
                # atom_stereo_count=molecule_data.get('atom_stereo_count'),
                # bond_stereo_count=molecule_data.get('bond_stereo_count'),
                # covalent_unit_count=molecule_data.get('covalent_unit_count'),
                # h_bond_acceptor_count=molecule_data.get('h_bond_acceptor_count'),
                # h_bond_donor_count=molecule_data.get('h_bond_donor_count'),
                # heavy_atom_count=molecule_data.get('heavy_atom_count'),
                # rotatable_bond_count=molecule_data.get('rotatable_bond_count'),
                # undefined_atom_stereo_count=molecule_data.get('undefined_atom_stereo_count'),
                # elements=molecule_data.get('elements'),
                # cactvs_fingerprint=molecule_data.get('cactvs_fingerprint'),
            )
            data_list.append(data)
        all_data = self.collate(data_list)
        torch.save(all_data, self.processed_paths[0])


class MolecularGraphsGDSC(InMemoryDataset, ABC):
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    stereo = {BS.STEREONONE: 0, BS.STEREOANY: 1, BS.STEREOZ: 2,
              BS.STEREOE: 3, BS.STEREOCIS: 4, BS.STEREOTRANS: 5}
    direction = {BD.NONE: 0, BD.BEGINWEDGE: 1, BD.BEGINDASH: 2,
                 BD.ENDDOWNRIGHT: 3, BD.ENDUPRIGHT: 4, BD.EITHERDOUBLE: 5,
                 BD.UNKNOWN: 6}

    def __init__(self, root):
        super().__init__(root)
        self.root = Path(root)
        # print(self.root)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        # print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0]) 

    @property
    def raw_dir(self):
        return Path(self.root) / 'data/raw/GDSC2'

    @property
    def processed_dir(self):
        return Path(self.root) / 'data/processed/pytorch_graphs/GDSC2'

    @property
    def raw_file_names(self):
        return "GDSC2_IC50_PUBCHEM.pkl"

    @property
    def processed_file_names(self):
        return "gdsc_molecular_graphs.pt"

    def process(self):
        """ Load data """
        labels = pd.read_pickle(self.raw_paths[0])
        fdef_name = Path(Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))

        molecules = list(labels['pubchem_id'].unique())

        molecule_list = []
        names = []
        data_list = []

        print("Creating molecular graphs")
        for molecule in enumerate(tqdm((molecules),
                                       total=len(molecules),
                                       position=0,
                                       leave=True)):
            pubchem_id = molecule[1]
            smiles = labels.loc[labels['pubchem_id'] ==
                                pubchem_id]['smiles']
            mol = Chem.MolFromSmiles(smiles.values[0])  # there are multiple returned values
            N = mol.GetNumAtoms()

            """ Features """
            atomic_number = []
            aromatic = []
            donor = []
            acceptor = []
            s = []
            sp = []
            sp2 = []
            sp3 = []
            sp3d = []
            sp3d2 = []
            num_hs = []

            for atom in mol.GetAtoms():
                # type_idx.append(self.types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                donor.append(0)
                acceptor.append(0)
                s.append(1 if hybridization == HybridizationType.S
                         else 0)
                sp.append(1 if hybridization == HybridizationType.SP
                          else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2
                           else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3
                           else 0)
                sp3d.append(1 if hybridization == HybridizationType.SP3D
                            else 0)
                sp3d2.append(1 if hybridization == HybridizationType.SP3D2
                             else 0)

                num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))

            feats = factory.GetFeaturesForMol(mol)
            for j in range(0, len(feats)):
                if feats[j].GetFamily() == 'Donor':
                    node_list = feats[j].GetAtomIds()
                    for k in node_list:
                        donor[k] = 1
                elif feats[j].GetFamily() == 'Acceptor':
                    node_list = feats[j].GetAtomIds()
                    for k in node_list:
                        acceptor[k] = 1

            x = torch.tensor([atomic_number,
                              acceptor,
                              donor,
                              aromatic,
                              s, sp, sp2, sp3, sp3d, sp3d2,
                              num_hs],
                             dtype=torch.float).t().contiguous()

            row, col, bond_idx, bond_stereo, bond_dir = [], [], [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                # bond_idx += 2 * [self.bonds[bond.GetBondType()]]
                # bond_stereo += 2 * [self.stereo[bond.GetStereo()]]
                # bond_dir += 2 * [self.direction[bond.GetBondDir()]]
                # 2* list, because the bonds are defined 2 times, start -> end,
                # and end -> start
            edge_index = torch.tensor([row, col], dtype=torch.long)

            """ Create distance matrix """
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=1000)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except:
                AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            pos_matrix = np.array(
                [[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                 for k in range(mol.GetNumAtoms())])
            dist_matrix = pairwise_distances(pos_matrix)

            adj_matrix = np.eye(mol.GetNumAtoms())
            for bond in mol.GetBonds():
                begin_atom = bond.GetBeginAtom().GetIdx()
                end_atom = bond.GetEndAtom().GetIdx()
                adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

            keys = ['x', 'pubchem_id', 'distance_matrix', 'adj_matrix', 'edge_index']
            values = [x, pubchem_id, dist_matrix, adj_matrix, edge_index]
            molecule_dict = dict(zip(keys, values))
            molecule_list.append(molecule_dict)
            names.append(pubchem_id)

        print("Concatenating the dataset")
        for molecule_data in molecule_list:
            pubchem_id = molecule_data.get("pubchem_id")
            x = molecule_data.get("x")
            distance_matrix = molecule_data.get("distance_matrix")
            adj_matrix = molecule_data.get("adj_matrix")
            edge_index = molecule_data.get('edge_index')

            data = MolData(
                x=x,
                adj_matrix=adj_matrix,
                dist_matrix=distance_matrix,
                pubchem_id=pubchem_id,
                edge_index=edge_index
            )
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])

class MolecularGraphsCTRP(InMemoryDataset, ABC):
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    stereo = {BS.STEREONONE: 0, BS.STEREOANY: 1, BS.STEREOZ: 2,
              BS.STEREOE: 3, BS.STEREOCIS: 4, BS.STEREOTRANS: 5}
    direction = {BD.NONE: 0, BD.BEGINWEDGE: 1, BD.BEGINDASH: 2,
                 BD.ENDDOWNRIGHT: 3, BD.ENDUPRIGHT: 4, BD.EITHERDOUBLE: 5,
                 BD.UNKNOWN: 6}

    def __init__(self, root):
        super().__init__(root)
        self.root = Path(root)
        # torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])
        # print(self.data, self.slices)

    @property
    def raw_dir(self):
        return Path(self.root) / 'data/raw/CTRP'

    @property
    def processed_dir(self):
        return Path(self.root) / 'data/processed/pytorch_graphs/CTRP'

    @property
    def raw_file_names(self):
        return "CTRP_all_v2.pkl"

    @property
    def processed_file_names(self):
        return "ctrpv2_molecular_graphs_public.pt"

    def atoms_features(self, mol, feats):
        """ Features """

        atomic_number, aromatic, donor, acceptor, s  = [], [], [], [], []
        sp, sp2, sp3, sp3d, sp3d2, num_hs = [], [], [], [], [], []
        for atom in mol.GetAtoms():
            # type_idx.append(self.types[atom.GetSymbol()])
            atomic_number.append(atom.GetAtomicNum())
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybridization = atom.GetHybridization()
            donor.append(0)
            acceptor.append(0)
            s.append(1 if hybridization == HybridizationType.S
                    else 0)
            sp.append(1 if hybridization == HybridizationType.SP
                    else 0)
            sp2.append(1 if hybridization == HybridizationType.SP2
                    else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3
                    else 0)
            sp3d.append(1 if hybridization == HybridizationType.SP3D
                        else 0)
            sp3d2.append(1 if hybridization == HybridizationType.SP3D2
                        else 0)

            num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))

        for j in range(0, len(feats)):
            if feats[j].GetFamily() == 'Donor':
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    donor[k] = 1
            elif feats[j].GetFamily() == 'Acceptor':
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    acceptor[k] = 1
        # return the features of nodes in molecular dataset
        x = torch.tensor([atomic_number,
                            acceptor,
                            donor,
                            aromatic,
                            s, sp, sp2, sp3, sp3d, sp3d2,
                            num_hs],
                            dtype=torch.float).t().contiguous()
        return x

    def edges_indexes(self, mol):
        row, col, bond_idx, bond_stereo, bond_dir = [], [], [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            
        return torch.tensor([row, col], dtype=torch.long)


    def other_feature_names(self):
        with open('/workspace/data/raw/desc.json', 'r', encoding='utf-8') as f:
            desc = json.load(f)

        return desc['rf_importance_features']


    def dump_to_json(self, data, names):
        with open("tmp_molecules.json", "w") as final:
            json.dump(data, final)
        with open("tmp_names.json", "w") as final:
            json.dump(names, final)
        return


    def read_from_json(self):
        names = json.load("tmp/names.json")
        return json.load("tmp/molecules.json"), names, len(names)


    def process(self):
        """ Load data """
        # bad_cids = [117560, 23939, 0, 23976]  # don't save these molecules
        labels = pd.read_pickle(self.raw_paths[0])
        labels = labels.groupby('pubchem_id', as_index=False).first()
        # labels = labels.loc[~labels['pubchem_id'].isin(bad_cids)]
        fdef_name = Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef'
        factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))

        molecules = list(labels['pubchem_id'].unique())
        molecule_list = []
        names = []
        data_list = []
        # other_feature_names = self.other_feature_names()
        print("Creating molecular graphs")
        for molecule in enumerate(tqdm((molecules),
                                       total=len(molecules),
                                       position=0,
                                       leave=True)):
            pubchem_id = molecule[1]
            smiles = labels.loc[labels['pubchem_id'] == pubchem_id]['isomeric_smiles']
            # other_features = labels.loc[labels['pubchem_id'] == name][other_feature_names].values[0]

            mol = Chem.MolFromSmiles(smiles.values[0])  # there are multiple returned values
            """ Features """
            try:
                feats = factory.GetFeaturesForMol(mol)
                x = self.atoms_features(mol, feats)
                edge_index = self.edges_indexes(mol)

                try:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, maxAttempts=5000)
                    AllChem.UFFOptimizeMolecule(mol)
                    mol = Chem.RemoveHs(mol)
                except:
                    AllChem.Compute2DCoords(mol)

                conf = mol.GetConformer()
                pos_matrix = np.array(
                    [[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                    for k in range(mol.GetNumAtoms())])
                dist_matrix = pairwise_distances(pos_matrix)

                """ Create adjecency matrix """
                adj_matrix = np.eye(mol.GetNumAtoms())
                for bond in mol.GetBonds():
                    begin_atom = bond.GetBeginAtom().GetIdx()
                    end_atom = bond.GetEndAtom().GetIdx()
                    adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

                keys = ['x', 'pubchem_id', 'distance_matrix', 'adj_matrix', 'edge_index']
                values = [x, pubchem_id, dist_matrix, adj_matrix, edge_index]
                molecule_dict = dict(zip(keys, values))
                molecule_list.append(molecule_dict)
                names.append(pubchem_id)

            except:
                print("name: {}, has a bug".format(names[-1]))

        print("Concatenating the dataset\n\n")
        for molecule_data in molecule_list:
            pubchem_id = molecule_data.get("pubchem_id")
            x = molecule_data.get("x")
            distance_matrix = molecule_data.get("distance_matrix")
            adj_matrix = molecule_data.get("adj_matrix")
            edge_index = molecule_data.get("edge_index")

            # Omit the targets one molecule has many different targets depending on the cell line
            data = MolData(
                x=x,
                edge_index=edge_index,
                adj_matrix=adj_matrix,
                dist_matrix=distance_matrix,
                pubchem_id = pubchem_id,
            )
            data_list.append(data)
        all_data = self.collate(data_list)
        torch.save(all_data, self.processed_paths[0])


if __name__ == '__main__':
    k= MolecularGraphs(root='/workspace')
    print(k[0])
"""    for i in range(0,len(k)):
        if k[i].pubchem_id == 407018 or k[i].pubchem_id == '407018':
            print('That molecule exists: ', k[i])
        print(k[i].pubchem_id)"""
    
  