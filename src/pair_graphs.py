from abc import ABC
from cProfile import label
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import BondStereo as BS
from rdkit.Chem.rdchem import BondDir as BD
from pathlib import Path
import torch
import json
from torch_geometric.data import Batch
from sklearn.metrics import pairwise_distances
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
from create_graphs_mol import (MolecularGraphs,
                           MolecularGraphsGDSC, MolData, 
                           MolecularGraphsCTRP)
from create_graphs_ppi import (PPIGraphsL1000,PPIGraphsDRP)


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, labels_path, problem: str = '', self_att: str = '', test_5000: bool = False):
        """labels_path = /workspace/data/processed/NCI60_blind"""
        self.problem=problem
        self.root = Path(__file__).resolve().parents[1].absolute()
        if 'GDSC' in str(labels_path):
            self.molecular_graphs = MolecularGraphsGDSC(self.root)
        elif 'NCI60' in str(labels_path):
            self.molecular_graphs = MolecularGraphs(self.root)
        elif 'CTRP' in str(labels_path):
            self.molecular_graphs = MolecularGraphsCTRP(self.root)
            print(self.molecular_graphs[0])
            
        if 'DRP' in str(labels_path):
            self.ppi_graphs = PPIGraphsDRP(self.root, self_att=self_att)
        else:
            self.ppi_graphs = PPIGraphsL1000(self.root)
        
        print('Pair db labels path: {} \n'.format(labels_path))
        try:
            self.labels_split = pd.read_csv(labels_path)[['pubchem_id', 
            'RRID', 'LN_IC50']]
            
        except:
            self.labels_split = labels_path[['pubchem_id', 'RRID', 'LN_IC50']]
        
        if self.problem == 'classification':
            # TODO is it a negative log value of uM concentration of compound OR 
            # is it just ln value of 1 uM = 1e-6 the value should be 
            # smaller than np.log(1) for sensitive and gdsc reports LN_IC50 
            # NCI60 LN_IC50 was calculated within data preparation notebooks.
            self.labels_split['sensitive_uM'] = np.where(self.labels_split['LN_IC50'] < 0, 1, 0)
            self.targets = self.labels_split['sensitive_uM']
            
        else: 
            self.targets = self.labels_split['LN_IC50']
            self.labels_split['sensitive_uM'] = self.labels_split['LN_IC50']
        # print(self.labels_split.info())
        # get indices with pubchem cids
        mol_index = []
        pubchem_cid = []
        i = 0
        if test_5000:
            with open('/workspace/data/raw/desc.json', 'r', encoding='utf-8') as f:
                desc = json.load(f)
            
            self.mol_indices = pd.Series(data=[i for i in range(len(desc['test_5000']))], index=desc['test_5000'])
            self.labels_split = self.labels_split[self.labels_split['pubchem_id'].isin(desc['test_5000'])]
            self.targets = self.labels_split['LN_IC50']
            # print('ls ok!')
        else:
            for graph in self.molecular_graphs:
                mol_index.append(i)
                pubchem_cid.append(graph.pubchem_id)
                # if graph.pubchem_id == 25262792:
                    # print(graph)
                i += 1
                
            self.mol_indices = pd.Series(data=mol_index, index=pubchem_cid)
        
        # print(self.labels_split[self.labels_split['pubchem_id']==25262792] )
        # get indices with cellosaurus accessions
        cell_index = []
        cellosaurus_accession = []
        i = 0
        for ppi_graph in self.ppi_graphs:
            cell_index.append(i)
            cellosaurus_accession.append(ppi_graph.cell_name)
            i += 1
        self.cell_indices = pd.Series(data=cell_index, 
                                      index=cellosaurus_accession)
        
    #     # print(self.mol_indices.loc[25262792])
    #     mi = set(self.mol_indices.index.values)
    #     k=0
    #     for i in self.labels_split.pubchem_id.unique():
    #         if i not in self.mol_indices.index:
    #             k+=1
    #     #print('koliko ih nema par jer nedostaje u molekulski graf: ', k)
    #     k=0
    #     for i in self.mol_indices.index:
    #         if i not in self.labels_split.pubchem_id:
    #             k+=1
    #    # print('koliko ih nema par jer nedostaje oznakama: ', k)
 
    def __getitem__(self, idx):
        item = self.labels_split.iloc[idx]
        mol_graph_idx = self.mol_indices.loc[item['pubchem_id']]
        ppi_graph_idx = self.cell_indices[item['RRID']]
        return self.molecular_graphs[
            mol_graph_idx.tolist()], self.ppi_graphs[
            ppi_graph_idx.tolist()], item['sensitive_uM']

    def __len__(self):
        return len(self.labels_split)

    def balance_sampler(self):
        sensitive = self.labels_split['sensitive_uM'].value_counts()[1]
        resistant = self.labels_split['sensitive_uM'].value_counts()[0]
        return [resistant, sensitive]


def collate_batch(data_list):
    """Batch two different types of data together"""
    list_adj_mat = []
    list_dist_mat = []
    list_node_feat = []
    max_shape = max([data[0].dist_matrix.shape[0] for data in data_list])
    for data in data_list:
        list_adj_mat.append(pad_array(data[0].adj_matrix, (max_shape, max_shape)))
        list_dist_mat.append(pad_array(data[0].dist_matrix, (max_shape, max_shape)))
        list_node_feat.append(pad_array(data[0].x, (max_shape, data[0].x.shape[1])))

    batchA = [torch.Tensor(features) for features in (np.array(list_adj_mat), 
                                                      np.array(list_dist_mat), 
                                                      np.array(list_node_feat))]
    batchB = Batch.from_data_list([data[1] for data in data_list])
    try:
        target = torch.Tensor([data[2] for data in data_list])
        return batchA, batchB, target
    except:
        return batchA, batchB

class SimpleCustomBatch:
    def __init__(self, data_list):
        """Batch two different types of data together"""
        list_adj_mat = []
        list_dist_mat = []
        list_node_feat = []
        max_shape = max([data[0].dist_matrix.shape[0] for data in data_list])
        for data in data_list:
            list_adj_mat.append(pad_array(data[0].adj_matrix, (max_shape, max_shape)))
            list_dist_mat.append(pad_array(data[0].dist_matrix, (max_shape, max_shape)))
            list_node_feat.append(pad_array(data[0].x, (max_shape, data[0].x.shape[1])))        
        self.batchA = [torch.Tensor(features) for features in (np.array(list_adj_mat), 
                                                        np.array(list_dist_mat), 
                                                        np.array(list_node_feat))]
        self.batchB = Batch.from_data_list([data[1] for data in data_list])
        self.target = torch.Tensor([data[2] for data in data_list])

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.batchA = [self.batchA[i].pin_memory() for i in range(len(self.batchA))]
        self.batchB = self.batchB.pin_memory()
        self.target = self.target.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

def pad_array(array, shape, dtype=np.float32):
    """Pad 2D input molecular arrays to same size"""
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


class PairDatasetBenchmark(torch.utils.data.Dataset):
    def __init__(self, labels_path, problem: str ='regression', self_att: str = ''):
        self.problem=problem
        # print("\n\n",self.problem,"   In Pair dataset benchmark!! \n ")
        self.root = Path(__file__).resolve().parents[1].absolute()
        # print(self.root)
        if 'DRP' in str(labels_path):
            self.molecular_graphs = MolecularGraphs(self.root )
            self.ppi_graphs = PPIGraphsDRP(self.root, self_att=self_att )
        else:    
            self.molecular_graphs = MolecularGraphsGDSC(self.root )
            self.ppi_graphs = PPIGraphsL1000(self.root, self_att=self_att )
        self.labels = pd.read_csv(labels_path)[['pubchem_id', 'RRID', 'LN_IC50']]
        if problem == 'classification':
            self.labels['sensitive_uM'] = np.where(self.labels['LN_IC50'] < 0, 1, 0)
            self.targets = self.labels['sensitive_uM']
            #self.labels['sensitive_uM'] = np.where(self.labels['LN_IC50'] > 1, 1, 0)
        else: 
            self.labels['sensitive_uM'] = self.labels['LN_IC50']
        # get indices with pubchem cids
        mol_index = []
        pubchem_cid = []
        i = 0
        for graph in self.molecular_graphs:
            mol_index.append(i)
            pubchem_cid.append(graph.pubchem_id)
            i += 1
        self.mol_indices = pd.Series(data=mol_index, index=pubchem_cid)

        # get indices with cellosaurus accessions
        cell_index = []
        cellosaurus_accession = []
        i = 0
        for ppi_graph in self.ppi_graphs:
            cell_index.append(i)
            cellosaurus_accession.append(ppi_graph.cell_name)
            i += 1
        self.cell_indices = pd.Series(data=cell_index, index=cellosaurus_accession)

    def __getitem__(self, idx):
        item = self.labels.iloc[idx]
        mol_graph_idx = self.mol_indices[(item['pubchem_id'])]
        ppi_graph_idx = self.cell_indices[item['RRID']]
        return self.molecular_graphs[mol_graph_idx.tolist()], self.ppi_graphs[ppi_graph_idx.tolist()], item['sensitive_uM']

    def __len__(self):
        return len(self.labels)

"""Dataset creator for creating datasets for inference"""


class InferenceDataset(torch.utils.data.Dataset):
    """PPI graphs are read from the torch PPI graphs file, while mol graphs are created"""
    def __init__(self, drug_id, smiles, cellosaurus_accession, dataset= 'GDSC'):
        self.labels = (pd.DataFrame({'smiles': smiles,
                                     'RRID': cellosaurus_accession,
                                     'drug_id': drug_id}))
        self.root = Path(__file__).resolve().parents[1].absolute()
        if dataset == 'NCI60DRP':
            self.ppi_graphs = PPIGraphsDRP(self.root)
        else:
            self.ppi_graphs = PPIGraphsL1000(self.root)

        # get indices with cellosaurus accessions
        cell_index = []
        cellosaurus_accession = []
        i = 0
        for ppi_graph in self.ppi_graphs:
            cell_index.append(i)
            cellosaurus_accession.append(ppi_graph.cell_name)
            i += 1
        self.cell_indices = pd.Series(data=cell_index, index=cellosaurus_accession)

        self.molecular_graphs = []
        mol_index = []
        drug_ids = []
        i = 0
        for index, mol_graph in self.labels.iterrows():
            self.molecular_graphs.append(self.make_mol_graph(mol_graph['smiles']))
            drug_ids.append(mol_graph['drug_id'])
            mol_index.append(i)
            i += 1
        self.molecular_indices = pd.Series(data=mol_index, index=drug_ids)

    def __getitem__(self, idx):
        item = self.labels.iloc[idx]
        ppi_graph_idx = self.cell_indices[item['RRID']]
        mol_graph_idx = self.molecular_indices[item['drug_id']]
        return self.molecular_graphs[mol_graph_idx.tolist()], self.ppi_graphs[ppi_graph_idx.tolist()]

    def __len__(self):
        return len(self.labels)

    def make_mol_graph(self, smiles):
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        stereo = {BS.STEREONONE: 0, BS.STEREOANY: 1, BS.STEREOZ: 2,
                  BS.STEREOE: 3, BS.STEREOCIS: 4, BS.STEREOTRANS: 5}
        direction = {BD.NONE: 0, BD.BEGINWEDGE: 1, BD.BEGINDASH: 2,
                     BD.ENDDOWNRIGHT: 3, BD.ENDUPRIGHT: 4, BD.EITHERDOUBLE: 5,
                     BD.UNKNOWN: 6}

        fdef_name = Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef'
        factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))
        mol = Chem.MolFromSmiles(smiles)
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
            bond_idx += 2 * [bonds[bond.GetBondType()]]
            bond_stereo += 2 * [stereo[bond.GetStereo()]]
            bond_dir += 2 * [direction[bond.GetBondDir()]]
            # 2* list, because the bonds are defined 2 times, start -> end,
            # and end -> start

        """ Create distance matrix """
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

        adj_matrix = np.eye(mol.GetNumAtoms())
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom().GetIdx()
            end_atom = bond.GetEndAtom().GetIdx()
            adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

        data = MolData(
            x=x,
            adj_matrix=adj_matrix,
            dist_matrix=dist_matrix,
        )

        return data

if __name__ == '__main__':
    # MolecularGraphs('/workspace')
    pass
