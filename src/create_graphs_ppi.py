import os
from abc import ABC
from multiprocessing import popen_fork
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
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset, Batch
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import pairwise_distances
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
import json
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

class PPIGraphsDRP(InMemoryDataset, ABC):
    def __init__(self, root, year: str = '22', 
                 use_pq: bool = False, 
                 self_att: str = '', 
                 ignore_cache: bool = False,
                 new_data: str = ''):
        
        self.new_data = new_data
        self.ignore_cache = ignore_cache
        self.root = root
        self.use_pq = use_pq
        self.self_att = self_att
        self.year = year
        if self.year == '20':
            self.pv = 0
        else: self.pv = 5
        # Coment line below for "Dataset" super() class
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_dir(self):
        return Path(self.root)

    @property
    def processed_dir(self):
        os.makedirs(self.raw_dir / 'data/processed/pytorch_graphs/PPIDRP{}/'.format(
            self.new_data), exist_ok=True)
        return self.raw_dir / 'data/processed/pytorch_graphs/PPIDRP{}/'.format(
            self.new_data)

    @property
    def raw_file_names(self):
        return ["depmap22q2/CCLE_expression.csv", "depmap22q2/CCLE_gene_cn.csv", 
        "depmap22q2/CCLE_mutations.csv", "depmap22q2/sample_info.csv", 
        "gygi_lab/protein_quant_current_normalized.csv", 'STRING/9606.protein.links.full.v11.5.txt',
        'STRING/9606.protein.info.v11.5.txt']

    @property
    def processed_file_names(self):
        return 'ppi_graphs{}.pt'.format(self.self_att)
        # For "Dataset" super() class use list below example: 
        # return ['ppi_graph_1.pt', 'ppi_graph_2.pt', ...]

    def save_oe(self, oe, genes, cell=''):
        with open(self.raw_dir / 'data/processed/pytorch_graphs/PPIDRP{}/oe{}.npy'.format(
            self.new_data, cell), 'wb') as f:
            np.save(f, np.concatenate((oe.T ,genes.T)))
        return
    def get_node_name(self, k, cell=''):
        indexes = np.load(self.raw_dir / 'data/processed/pytorch_graphs/PPIDRP{}/oe{}.npy'.format(
            self.new_data, cell), allow_pickle=True)
        return indexes[1][k]
    
    def get_node_index(self, k):
        return np.load(self.raw_dir / 'data/processed/pytorch_graphs/PPIDRPApr/oe{}.npy'.format(
            self.new_data,cell), allow_pickle=True)
    
    # For "Dataset" super() class use list below example:
    #def get(self, idx):
        #data = torch.load(self.processed_dir / f'ppi_graph_{idx}.pt')
        #return data

    def process(self):
        print("Creating PPI graphs")
        project_dir = self.raw_dir
        d = 'data/processed/NCI60DRP{}_random{}/'.format( 
                        self.new_data, self.self_att)
        
        dir_path = project_dir / d
        if Path(dir_path / 'cell_features_drp{}.pkl'.format(
            self.self_att)).is_file() and not self.ignore_cache:
            expression = pd.read_pickle(dir_path / 
                        'cell_features_drp{}.pkl'.format(self.self_att))
        else:
            mutations = pd.read_csv(project_dir / 'data/raw/depmap22q2/CCLE_mutations.csv')
            expression = pd.read_csv(project_dir / 'data/raw/depmap22q2/CCLE_expression.csv', index_col=0)
            sample_info = pd.read_csv(project_dir / 'data/raw/depmap22q2/sample_info.csv')
            cnv = pd.read_csv(project_dir / 'data/raw/depmap22q2/CCLE_gene_cn.csv', index_col=0)
            selection = pd.read_csv(project_dir / 'data/raw/NCI60June2022/nci60_selection_marker_genes.csv')
            # landmark_genes = pd.read_csv(project_dir / 'data/raw/landmark_genes_l1000.txt', sep='\t')
            # landmark_genes_list = list(landmark_genes['pr_gene_symbol'])
        
            # edit expression data
            # landmark_genes_list = list(landmark_genes['pr_gene_symbol'])
            expression.columns = expression.columns.str.split(' ').str[0]
            # cols = [col for col in expression.columns if col in landmark_genes_list]  # TPM of landmark genes only
            # expression = expression[cols]
            expression = expression.loc[expression.index.isin(selection.depmap_id.unique())]
            expression.reset_index(inplace=True)
            expression.rename(columns={'index': 'DepMap_ID'}, inplace=True)
            expression = expression.merge(sample_info[['DepMap_ID', 'RRID']], how='inner', on='DepMap_ID')
            # expression.rename(columns={'RRID': 'cellosaurus_accession'}, inplace=True)
            expression = pd.melt(expression, id_vars=['RRID', 'DepMap_ID'], var_name='gene',
                                value_name='tpm')
            expression.dropna(inplace=True)

            # edit copy number variation data
            cnv.columns = cnv.columns.str.split(' ').str[0]
            # cnv_cols = [col for col in cnv.columns if col in landmark_genes_list]
            # cnv = cnv[cnv_cols]
            cnv.reset_index(inplace=True)
            cnv.rename(columns={'index': 'DepMap_ID'}, inplace=True)
            cnv = cnv.merge(sample_info[['DepMap_ID', 'RRID']], how='inner', on='DepMap_ID')
            # cnv.rename(columns={'RRID': 'cellosaurus_accession'}, inplace=True)
            cnv = pd.melt(cnv, id_vars=['RRID', 'DepMap_ID'], var_name='gene',
                        value_name='cnv')
            cnv.dropna(inplace=True)

            expression = expression.merge(cnv[['RRID', 'gene', 'cnv']], 
                                          how='inner', on=['RRID', 'gene'])
            
            dependency = pd.read_csv('/workspace/data/raw/NCI60June2022/nci60_selection_28_Apr.csv')
            dependency = pd.merge(dependency, selection[['cellosaurus_id', 'depmap_id']], how='left', on='depmap_id')
            dependency.rename(columns={
                                'cellosaurus_id': 'RRID',
                                'gene_name':'gene'
                                }, inplace=True)
            
            expression = expression.merge(dependency[['RRID', 'gene', 'dependency']], 
                                                    how='inner', on=['RRID', 'gene'])
            
                        # edit mutation data from CCLE (DB)
            mutations.rename(columns={'Hugo_Symbol': 'gene'}, inplace=True)
            expression = expression.merge(mutations[['gene', 'DepMap_ID', 
                                        'Variant_Classification', 'Variant_Type']],
                                        how='left', left_on=['DepMap_ID', 'gene'], 
                                        right_on=['DepMap_ID', 'gene'])
            
            expression['Variant_Classification'] = expression['Variant_Classification'].fillna(value='Wild_Type')
            expression['Variant_Type'] = expression['Variant_Type'].fillna(value='WT')
            expression.drop_duplicates(subset=['RRID', 'gene'], inplace=True)
            
            # Ordinal encoding variant types
            variant_type_encoder = OrdinalEncoder()
            variant_type_or = variant_type_encoder.fit_transform(
                expression[['Variant_Type']])
            
            variant_classification_encoder = OrdinalEncoder()
            variant_classification_or = variant_classification_encoder.fit_transform(
                expression[['Variant_Type']])
            
            expression['variant_type_or'] = variant_type_or.reshape(-1)
            expression['variant_classification_or'] = variant_classification_or.reshape(-1)
            cnv=[]
            mutations=[]
            expression.to_pickle(project_dir / 'data/processed/cell_features_drp.pkl')

        # edit PPI network data from STRING (DB)
        if Path(dir_path / 'ppi_links_drp{}.pkl'.format(
                self.self_att)).is_file() and not self.ignore_cache:
            ppi_links_cell = pd.read_pickle(dir_path / 
                            'ppi_links_drp{}.pkl'.format(self.self_att))
            
        else:
            ppi_links = pd.read_csv(project_dir / 'data/raw/STRING/9606.protein.links.full.v11.5.txt', delim_whitespace=True)
            ppi_info = pd.read_csv(project_dir / 'data/raw/STRING/9606.protein.info.v11.5.txt', sep='\t')
            ppi_links = ppi_links.loc[ppi_links['experiments'] != 0][['protein1', 'protein2', 'combined_score']]
            ppi_links = ppi_links.merge(ppi_info[['#string_protein_id', 'preferred_name']], how='left',
                            left_on='protein1', right_on='#string_protein_id')
            ppi_links = ppi_links.merge(ppi_info[['#string_protein_id', 'preferred_name']], how='left',
                            left_on='protein2', right_on='#string_protein_id')
            ppi_links.drop(columns=['#string_protein_id_x', '#string_protein_id_y', 'protein1', 'protein2'],
                inplace=True)

            """ further reduce the dimensionality of the dataset"""
            ppi_links.rename(columns={
                'preferred_name_x': 'protein_1',
                'preferred_name_y': 'protein_2',
            }, inplace=True)
            ppi_links_cell = ppi_links.loc[(ppi_links['protein_1'].isin(expression['gene'])) &
                                        (ppi_links['protein_2'].isin(expression['gene']))]
            ppi_links_cell.to_pickle(project_dir / 'data/processed/NCI60DRP{}_random{}/ppi_links_drp{}.pkl'.format(
                self.new_data, self.self_att, self.self_att))
            print(' expression shape{}'.format(expression['gene'].unique().shape), 
                  ' ppi_links_protein shape{}'.format(ppi_links_cell['protein_1'].unique().shape),
                  ' vs ppi links: {}'.format(ppi_links['protein_1'].unique().shape))
            ppi_links = []
            ppi_info = []
        # add mass spectrometry quantities of protein Gygi Lab

        if (Path(dir_path / 'p_quantities_drp{}.pkl'.format(
            self.self_att)).is_file() and self.use_pq
            ):
            p_quantities = pd.read_pickle(dir_path / 
                'p_quantities_drp{}.pkl'.format(self.self_att))

        elif self.use_pq:
            sample_info = pd.read_csv(project_dir / 'data/raw/depmap22q2/sample_info.csv')
            p_quantities = pd.read_csv(project_dir / 'data/raw/gygi_lab/protein_quant_current_normalized.csv')
            protein = p_quantities['Protein_Id'].str.strip('_HUMAN')
            p_quantities['protein'] = protein.str.split('|').str[2]
            p_quantities = p_quantities.drop(['Protein_Id','Description','Group_ID'], axis=1)
            
            # drop peptides
            p_quantities = p_quantities.drop(p_quantities.columns[3:45], axis=1)
            p_quantities = pd.melt(p_quantities, id_vars=['Uniprot', 'Gene_Symbol', 'Uniprot_Acc', 'protein'], var_name='Name_Tissue',
                        value_name='quantity')
            p_quantities.Name_Tissue = p_quantities.Name_Tissue.replace(to_replace ='_TenPx*', value = '', regex = True)
            p_quantities['Tissue'] = p_quantities.Name_Tissue.str.extract(r'((?<=_).*$)', expand=True)
            p_quantities['stripped_cell_line_name'] = p_quantities.Name_Tissue.str.extract(r'(.+?)_', expand=True)
            p_quantities = p_quantities.drop(['Uniprot', 'Name_Tissue'], axis=1)
            p_quantities = p_quantities.merge(sample_info[['stripped_cell_line_name', 'DepMap_ID']], how='inner', on='stripped_cell_line_name')
            sample_info = []
            p_quantities.to_pickle(dir_path / 
                        'p_quantities_drp{}.pkl'.format(self.self_att))
        else:
            pass
        cells = expression['RRID'].unique()
        print("Creating PPI graphs", cells)
        data_list = []
        ppi=pd.DataFrame([])
        for cell in tqdm(cells[:], position=0, leave=True):
            idx = expression['RRID'] == cell
            cell_expression = expression.loc[idx]
            oe_gene = OrdinalEncoder()
            #cell_expression['cell_gene_ordinal'] = oe_gene.transform(expression.loc[idx,'gene'].values.reshape(-1,1))
            # Make the edge list (for adjecency matrix of each cell_line)
            if self.self_att == '_self_att':               
                idx_2 = (ppi_links_cell['protein_1'].isin(
                    cell_expression['gene'])) & (
                    ppi_links_cell['protein_2'].isin(
                        cell_expression['gene']))
                
                pi1 = ppi_links_cell[['protein_1', 'protein_2']].loc[idx_2]
                cell_expression = cell_expression.loc[
                    cell_expression['gene'].isin(pi1['protein_1'].unique())]
                
                oe_gene.fit(cell_expression['gene'].unique().reshape(-1,1))
                oe_length = cell_expression['gene'].unique().shape[0]
                oe_l = oe_gene.transform(
                    cell_expression['gene'].unique().reshape(-1,1))
                
                gene_l = cell_expression['gene'].unique().reshape(-1,1)
                cell_expression['cell_gene_ordinal'] = oe_gene.transform(
                    cell_expression['gene'].values.reshape(-1,1))
                
                self.save_oe(oe_l, gene_l, cell + self.self_att)
                #expression.loc[idx,'cell_gene_ordinal'] = oe_gene.transform(
                #    expression.loc[idx,'gene'].values.reshape(-1,1))
                
                ordinal_genes = oe_gene.transform(pi1.values.reshape(-1,1))
                ai = oe_gene.transform(pi1.protein_1.values.reshape(-1,1))
                # Adding self attention to nodes. 
                aj = oe_gene.transform(pi1.protein_2.values.reshape(-1,1))
                pi1 = pi1.assign(oe_protein1 = ai)
                pi1 = pi1.assign(oe_protein2 = aj)
                ai = np.concatenate((ai, oe_l), axis=0)
                aj = np.concatenate((aj, oe_l), axis=0)
                x = cell_expression.drop_duplicates(
                    subset=['cell_gene_ordinal'])
                
                ppi = pd.concat([ppi, pi1], ignore_index=True)
                
            else:
                idx_2 = (ppi_links_cell['protein_1'].isin(expression.loc[idx,'gene'])) & (
                    ppi_links_cell['protein_2'].isin(expression.loc[idx,'gene']))
                
                pi1 = ppi_links_cell.loc[idx_2, ['protein_1', 'protein_2']]
                # print(pi1.values.ravel())
                idx = cell_expression['gene'].isin(pi1.values.ravel())
                cell_expression = cell_expression[idx]
                oe_gene.fit(cell_expression['gene'].unique().reshape(-1,1))
                oe_length = cell_expression['gene'].unique().shape[0]
                oe_l = oe_gene.transform(cell_expression['gene'].unique().reshape(-1,1))
                gene_l = cell_expression['gene'].unique().reshape(-1,1)
                self.save_oe(oe_l, gene_l, cell + self.self_att)
                cell_expression.loc[idx,'cell_gene_ordinal'] = oe_gene.transform(
                    cell_expression.loc[idx,'gene'].values.reshape(-1,1))
                
                ai = oe_gene.transform(pi1.protein_1.values.reshape(-1,1))
                aj = oe_gene.transform(pi1.protein_2.values.reshape(-1,1))
                pi1 = pi1.assign(oe_protein1 = ai)
                pi1 = pi1.assign(oe_protein2 = aj)
                x = cell_expression.loc[idx].drop_duplicates(
                    subset=['cell_gene_ordinal'])
                
                #TODO add to expression.loc[expression['RRID'] == cell]
                ppi = pd.concat([ppi, pi1], ignore_index=True)
                
            edge_list_simple = torch.tensor(np.concatenate((ai.T, aj.T), axis=0), dtype=torch.long)
            # print(ordinal_genes)
            x = x.sort_values(by=['cell_gene_ordinal'])[['dependency', 'tpm','cnv',
                                            'variant_type_or','variant_classification_or']]
            
            x_ppi = torch.cat([torch.from_numpy(x['dependency'].values).unsqueeze(-1),
                                torch.from_numpy(x['tpm'].values).unsqueeze(-1),
                                torch.from_numpy(x['cnv'].values).unsqueeze(-1),
                                torch.from_numpy(x['variant_type_or'].values).unsqueeze(-1),
                                torch.from_numpy(x['variant_classification_or'].values).unsqueeze(-1),
                                ], dim=-1).to(torch.float)
            
            data = Data(x=x_ppi,
                    edge_index=edge_list_simple,
                    cell_name=cell,
                    )
            data_list.append(data)

            # Use for InMemoryDataset super() class
        torch.save(self.collate(data_list), self.processed_paths[0])
        ppi.to_pickle(dir_path / 'ppi_links_drp{}.pkl'.format(
            self.self_att))
        
        expression.to_pickle(dir_path / 'cell_features_drp{}.pkl'.format(
            self.self_att))
        
            # For Dataset super() class 
            # torch.save(data, project_dir / 'data/processed/pytorch_graphs/PPIDRP/ppi_graph_{}.pt'.format(idx))
            
            #except:
               # print('Failed at index {}'.format(idx))
               # idx = idx +1


class PPIGraphsL1000(InMemoryDataset, ABC):

    def __init__(self, root, year: str = '22', use_pq: bool = False, self_att: str = ''):
        self.root=root
        self.use_pq=use_pq
        self.year=year
        self.self_att = self_att

        if self.year == '20':
            self.pv = 0
        else: self.pv = 5
        super().__init__(root)
        # Coment line below for "Dataset" super() class
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_dir(self):
        return Path(self.root)


    @property
    def processed_dir(self):
        os.makedirs('/workspace/data/processed/pytorch_graphs/PPI_l1000/', exist_ok=True)
        return Path(self.root) / 'data/processed/pytorch_graphs/PPI_l1000/'


    @property
    def raw_file_names(self):
        return ["depmap22q2/CCLE_expression.csv", "depmap22q2/CCLE_gene_cn.csv", 
        "depmap22q2/CCLE_mutations.csv", "depmap22q2/sample_info.csv", 
        "gygi_lab/protein_quant_current_normalized.csv", 'STRING/9606.protein.links.full.v11.5.txt',
        'STRING/9606.protein.info.v11.5.txt']


    @property
    def processed_file_names(self):
        return 'ppi_graphs{}.pt'.format(self.self_att)

    def save_oe(self, oe, genes):
        os.makedirs('/workspace/data/processed/pytorch_graphs/PPI_l1000', exist_ok=True) 
        with open('/workspace/data/processed/pytorch_graphs/PPI_l1000/oe.npy', 'wb+') as f:
            np.save(f, np.concatenate((oe.T ,genes.T)))
        return

    def get_node_name(self, k):
        indexes = np.load('/workspace/data/processed/pytorch_graphs/PPI_l1000/oe.npy', allow_pickle=True)
        return indexes[1][k]


    def get_node_index(self, k):
        return np.load('/workspace/data/processed/pytorch_graphs/PPI_l1000/oe.npy', allow_pickle=True)


    def __len__(self):
        return len(self.data)

    def process(self):
        print("Creating PPI graphs!")
        project_dir = self.raw_dir
        if Path(project_dir / 'data/processed/cell_features_l1000.pkl').is_file():
            expression = pd.read_pickle(project_dir / 'data/processed/cell_features_l1000.pkl')
        else:
            mutations = pd.read_csv(project_dir / 'data/raw/depmap22q2/CCLE_mutations.csv')
            expression = pd.read_csv(project_dir / 'data/raw/depmap22q2/CCLE_expression.csv', index_col=0)
            sample_info = pd.read_csv(project_dir / 'data/raw/depmap22q2/sample_info.csv')
            cnv = pd.read_csv(project_dir / 'data/raw/depmap22q2/CCLE_gene_cn.csv', index_col=0)
            landmark_genes = pd.read_csv(project_dir / 'data/raw/landmark_genes_l1000.txt', sep='\t')
            landmark_genes_list = list(landmark_genes['pr_gene_symbol'])
        
            # edit expression data
            landmark_genes_list = list(landmark_genes[landmark_genes['pr_is_lm'] == 1 ].pr_gene_symbol)
            expression.columns = expression.columns.str.split(' ').str[0]
            cols = [col for col in expression.columns if col in landmark_genes_list]  # TPM of landmark genes only
            expression = expression[cols]
            expression.reset_index(inplace=True)
            expression.rename(columns={'index': 'DepMap_ID'}, inplace=True)
            expression = expression.merge(sample_info[['DepMap_ID', 'RRID']], how='inner', on='DepMap_ID')
            # expression.rename(columns={'RRID': 'cellosaurus_accession'}, inplace=True)
            expression = pd.melt(expression, id_vars=['RRID', 'DepMap_ID'], var_name='gene',
                                value_name='tpm')
            expression.dropna(inplace=True)

            # edit copy number variation data
            cnv.columns = cnv.columns.str.split(' ').str[0]
            cnv_cols = [col for col in cnv.columns if col in landmark_genes_list]
            cnv = cnv[cnv_cols]
            cnv.reset_index(inplace=True)
            cnv.rename(columns={'index': 'DepMap_ID'}, inplace=True)
            cnv = cnv.merge(sample_info[['DepMap_ID', 'RRID']], how='inner', on='DepMap_ID')
            # cnv.rename(columns={'RRID': 'cellosaurus_accession'}, inplace=True)
            cnv = pd.melt(cnv, id_vars=['RRID', 'DepMap_ID'], var_name='gene',
                        value_name='cnv')
            cnv.dropna(inplace=True)

            expression = expression.merge(cnv[['RRID', 'gene', 'cnv']], how='inner',
                                        on=['RRID', 'gene'])

            # edit mutation data from CCLE (DB)
            mutations.rename(columns={'Hugo_Symbol': 'gene'}, inplace=True)
            expression = expression.merge(mutations[['gene', 'DepMap_ID', 'Variant_Classification', 'Variant_Type']],
                                        how='left', left_on=['DepMap_ID', 'gene'], right_on=['DepMap_ID', 'gene'])
            expression['Variant_Classification'] = expression['Variant_Classification'].fillna(value='Wild_Type')
            expression['Variant_Type'] = expression['Variant_Type'].fillna(value='WT')
            expression.drop_duplicates(subset=['RRID', 'gene'], inplace=True)
            variant_type_encoder = OneHotEncoder(sparse=False)
            variant_type_oh = variant_type_encoder.fit_transform(expression[['Variant_Type']])
            expression['variant_type_oh'] = variant_type_oh.tolist()
            cnv=[]
            mutations=[]
            expression.to_pickle(project_dir / 'processed/cell_features_l1000.pkl')

        # edit PPI network data from STRING (DB)
        if Path(project_dir / 'data/processed/ppi_links_l1000.pkl').is_file():
            ppi_links_cell = pd.read_pickle(project_dir / 'data/processed/ppi_links_l1000.pkl')
        else:
            ppi_links = pd.read_csv(project_dir / 'data/raw/STRING/9606.protein.links.full.v11.5.txt', delim_whitespace=True)
            ppi_info = pd.read_csv(project_dir / 'data/raw/STRING/9606.protein.info.v11.5.txt', sep='\t')
            ppi_links = ppi_links.loc[ppi_links['experiments'] != 0][['protein1', 'protein2', 'combined_score']]
            ppi_links = ppi_links.merge(ppi_info[['#string_protein_id', 'preferred_name']], how='left',
                            left_on='protein1', right_on='#string_protein_id')
            ppi_links = ppi_links.merge(ppi_info[['#string_protein_id', 'preferred_name']], how='left',
                            left_on='protein2', right_on='#string_protein_id')
            ppi_links.drop(columns=['#string_protein_id_x', '#string_protein_id_y', 'protein1', 'protein2'],
                inplace=True)

            """ further reduce the dimensionality of the dataset"""
            ppi_links.rename(columns={
                'preferred_name_x': 'protein_1',
                'preferred_name_y': 'protein_2',
            }, inplace=True)
            ppi_links_cell = ppi_links.loc[(ppi_links['protein_1'].isin(expression['gene'])) &
                                        (ppi_links['protein_2'].isin(expression['gene']))]
            print(ppi_links_cell.info())
            ppi_links_cell.to_pickle(project_dir / 'data/processed/ppi_links_l1000.pkl')

            ppi_links = []
            ppi_info = []

        # add mass spectrometry quantities of protein Gygi Lab

        if (Path(project_dir / 'data/processed/p_quantities_l1000.pkl').is_file() and self.use_pq):
            p_quantities = pd.read_pickle(project_dir / 'data/processed/p_quantities_l1000.pkl')
        elif self.use_pq:
            sample_info = pd.read_csv(project_dir / 'data/raw/depmap22q2/sample_info.csv')
            p_quantities = pd.read_csv(project_dir / 'data/raw/gygi_lab/protein_quant_current_normalized.csv')
            protein = p_quantities['Protein_Id'].str.strip('_HUMAN')
            p_quantities['protein'] = protein.str.split('|').str[2]
            p_quantities = p_quantities.drop(['Protein_Id','Description','Group_ID'], axis=1)
            # drop peptides
            p_quantities = p_quantities.drop(p_quantities.columns[3:45], axis=1)
            p_quantities = pd.melt(p_quantities, id_vars=['Uniprot', 'Gene_Symbol', 'Uniprot_Acc', 'protein'], var_name='Name_Tissue',
                        value_name='quantity')
            p_quantities.Name_Tissue = p_quantities.Name_Tissue.replace(to_replace ='_TenPx*', value = '', regex = True)
            p_quantities['Tissue'] = p_quantities.Name_Tissue.str.extract(r'((?<=_).*$)', expand=True)
            p_quantities['stripped_cell_line_name'] = p_quantities.Name_Tissue.str.extract(r'(.+?)_', expand=True)
            p_quantities = p_quantities.drop(['Uniprot', 'Name_Tissue'], axis=1)
            p_quantities = p_quantities.merge(sample_info[['stripped_cell_line_name', 'DepMap_ID']], how='inner', on='stripped_cell_line_name')
            sample_info = []
            p_quantities.to_pickle(project_dir / 'data/processed/p_quantities_l1000.pkl')
        else:
            pass

        print("Creating PPI graphs")
        oe = OrdinalEncoder()
        oe.fit(expression['gene'].unique().reshape(-1,1))
        oe_length = expression['gene'].unique().shape[0]
        oe_l = oe.transform(expression['gene'].unique().reshape(-1,1))
        gene_l = expression['gene'].unique().reshape(-1,1)
        self.save_oe(oe_l, gene_l)
        ai = oe.transform(ppi_links_cell.protein_1.values.reshape(-1,1))
        aj = oe.transform(ppi_links_cell.protein_2.values.reshape(-1,1))
        if self.self_att:
            ai = np.concatenate((ai, oe_l), axis=0)
            aj = np.concatenate((aj, oe_l), axis=0)
            
        edge_list_simple = torch.tensor(np.concatenate((ai.T, aj.T), axis=0), dtype=torch.long)
        expression['gene_ordinal'] = oe.transform(expression['gene'].values.reshape(-1,1))
        node_name= oe.inverse_transform(np.array([i for i in range(oe_length)]).reshape(-1, 1))
        self.save_oe(np.array([i for i in range(oe_length)]).reshape(-1, 1), node_name)
        cells = expression['RRID'].unique()
        data_list = []
        idx = 0
        
        for cell in tqdm(cells, position=0, leave=True):
            #try:
            cell_expression = expression.loc[expression['RRID'] == cell]
            x = cell_expression.drop_duplicates(subset=['gene_ordinal'])
            x = x.sort_values(by=['gene_ordinal'])[['tpm','cnv','variant_type_oh']]
            x_ppi = torch.cat([torch.from_numpy(x['tpm'].values).unsqueeze(-1),
                                torch.from_numpy(x['cnv'].values).unsqueeze(-1),
                                torch.from_numpy(np.array([np.array(i) for i in x['variant_type_oh'].values])),
                                ],
                                dim=-1).to(torch.float)
            data = Data(x=x_ppi,
                    edge_index=edge_list_simple,
                    cell_name=cell)
            data_list.append(data)
            idx = idx +1
            #except:
            #    print('Failed at index {}'.format(idx))
            #    idx = idx +1

        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == '__main__':
    ppi = PPIGraphsDRP(root='/workspace', self_att='', new_data='Apr') # self_att = '_self_att'
    for i in range(0,len(ppi)):
        print(ppi.get(i))
""" 
class PPIGraphsL1000(InMemoryDataset, ABC):

    def __init__(self, root, year = '22', use_pq = False):
        self.root=root
        self.use_pq=use_pq
        self.year=year

        if self.year == '20':
            self.pv = 0
        else: self.pv = 5
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_dir(self):
        return Path(self.root)


    @property
    def processed_dir(self):
        os.makedirs('/workspace/data/processed/pytorch_graphs/PPI_l1000/', exist_ok=True)
        return Path(self.root) / 'data/processed/pytorch_graphs/PPI_l1000/'


    @property
    def raw_file_names(self):
        return ["depmap22q2/CCLE_expression.csv", "depmap22q2/CCLE_gene_cn.csv", 
        "depmap22q2/CCLE_mutations.csv", "depmap22q2/sample_info.csv", 
        "gygi_lab/protein_quant_current_normalized.csv", 'STRING/9606.protein.links.full.v11.5.txt',
        'STRING/9606.protein.info.v11.5.txt']


    @property
    def processed_file_names(self):
        return 'ppi_graphs.pt'

    def save_oe(self, oe, genes):
        os.makedirs('/workspace/data/processed/pytorch_graphs/PPI_l1000', exist_ok=True) 
        with open('/workspace/data/processed/pytorch_graphs/PPI_l1000/oe.npy', 'wb+') as f:
            np.save(f, np.concatenate((oe.T ,genes.T)))
        return

    def get_node_name(self, k):
        indexes = np.load('/workspace/data/processed/pytorch_graphs/PPI_l1000/oe.npy', allow_pickle=True)
        return indexes[1][k]


    def get_node_index(self, k):
        return np.load('/workspace/data/processed/pytorch_graphs/PPI_l1000/oe.npy', allow_pickle=True)


    def __len__(self):
        return len(self.data)

    def process(self):
        print("Creating PPI graphs")
        project_dir = self.raw_dir
        if Path(project_dir / 'processed/cell_features_l1000.pkl').is_file():
            expression = pd.read_pickle(project_dir / 'processed/cell_features_l1000.pkl')
        else:
            mutations = pd.read_csv(project_dir / 'data/raw/depmap22q2/CCLE_mutations.csv')
            expression = pd.read_csv(project_dir / 'data/raw/depmap22q2/CCLE_expression.csv', index_col=0)
            sample_info = pd.read_csv(project_dir / 'data/raw/depmap22q2/sample_info.csv')
            cnv = pd.read_csv(project_dir / 'data/raw/depmap22q2/CCLE_gene_cn.csv', index_col=0)
            landmark_genes = pd.read_csv(project_dir / 'data/raw/landmark_genes_l1000.txt', sep='\t')
            landmark_genes_list = list(landmark_genes['pr_gene_symbol'])
        
            # edit expression data
            landmark_genes_list = list(landmark_genes[landmark_genes['pr_is_lm'] == 1 ].pr_gene_symbol)
            expression.columns = expression.columns.str.split(' ').str[0]
            cols = [col for col in expression.columns if col in landmark_genes_list]  # TPM of landmark genes only
            expression = expression[cols]
            expression.reset_index(inplace=True)
            expression.rename(columns={'index': 'DepMap_ID'}, inplace=True)
            expression = expression.merge(sample_info[['DepMap_ID', 'RRID']], how='inner', on='DepMap_ID')
            expression.rename(columns={'RRID': 'cellosaurus_accession'}, inplace=True)
            expression = pd.melt(expression, id_vars=['cellosaurus_accession', 'DepMap_ID'], var_name='gene',
                                value_name='tpm')
            expression.dropna(inplace=True)

            # edit copy number variation data
            cnv.columns = cnv.columns.str.split(' ').str[0]
            cnv_cols = [col for col in cnv.columns if col in landmark_genes_list]
            cnv = cnv[cnv_cols]
            cnv.reset_index(inplace=True)
            cnv.rename(columns={'index': 'DepMap_ID'}, inplace=True)
            cnv = cnv.merge(sample_info[['DepMap_ID', 'RRID']], how='inner', on='DepMap_ID')
            cnv.rename(columns={'RRID': 'cellosaurus_accession'}, inplace=True)
            cnv = pd.melt(cnv, id_vars=['cellosaurus_accession', 'DepMap_ID'], var_name='gene',
                        value_name='cnv')
            cnv.dropna(inplace=True)

            expression = expression.merge(cnv[['cellosaurus_accession', 'gene', 'cnv']], how='inner',
                                        on=['cellosaurus_accession', 'gene'])

            # edit mutation data from CCLE (DB)
            mutations.rename(columns={'Hugo_Symbol': 'gene'}, inplace=True)
            expression = expression.merge(mutations[['gene', 'DepMap_ID', 'Variant_Classification', 'Variant_Type']],
                                        how='left', left_on=['DepMap_ID', 'gene'], right_on=['DepMap_ID', 'gene'])
            expression['Variant_Classification'] = expression['Variant_Classification'].fillna(value='Wild_Type')
            expression['Variant_Type'] = expression['Variant_Type'].fillna(value='WT')
            expression.drop_duplicates(subset=['cellosaurus_accession', 'gene'], inplace=True)
            variant_type_encoder = OneHotEncoder(sparse=False)
            variant_type_oh = variant_type_encoder.fit_transform(expression[['Variant_Type']])
            expression['variant_type_oh'] = variant_type_oh.tolist()
            cnv=[]
            mutations=[]
            expression.to_pickle(project_dir / 'processed/cell_features_l1000.pkl')

        # edit PPI network data from STRING (DB)
        if Path(project_dir / 'processed/ppi_links_l1000.pkl').is_file():
            ppi_links_cell = pd.read_pickle(project_dir / 'processed/ppi_links_l1000.pkl')
        else:
            ppi_links = pd.read_csv(project_dir / 'data/raw/STRING/9606.protein.links.full.v11.5.txt', delim_whitespace=True)
            ppi_info = pd.read_csv(project_dir / 'data/raw/STRING/9606.protein.info.v11.5.txt', sep='\t')
            ppi_links = ppi_links.loc[ppi_links['experiments'] != 0][['protein1', 'protein2', 'combined_score']]
            ppi_links = ppi_links.merge(ppi_info[['#string_protein_id', 'preferred_name']], how='left',
                            left_on='protein1', right_on='#string_protein_id')
            ppi_links = ppi_links.merge(ppi_info[['#string_protein_id', 'preferred_name']], how='left',
                            left_on='protein2', right_on='#string_protein_id')
            ppi_links.drop(columns=['#string_protein_id_x', '#string_protein_id_y', 'protein1', 'protein2'],
                inplace=True)

            #further reduce the dimensionality of the dataset
            
            ppi_links.rename(columns={
                'preferred_name_x': 'protein_1',
                'preferred_name_y': 'protein_2',
            }, inplace=True)
            ppi_links_cell = ppi_links.loc[(ppi_links['protein_1'].isin(expression['gene'])) &
                                        (ppi_links['protein_2'].isin(expression['gene']))]
            print(ppi_links_cell.info())
            ppi_links_cell.to_pickle(project_dir / 'processed/ppi_links_l1000.pkl')

            ppi_links = []
            ppi_info = []

        # add mass spectrometry quantities of protein Gygi Lab

        if (Path(project_dir / 'processed/p_quantities_l1000.pkl').is_file() and self.use_pq):
            p_quantities = pd.read_pickle(project_dir / 'processed/p_quantities_l1000.pkl')
        elif self.use_pq:
            sample_info = pd.read_csv(project_dir / 'data/raw/depmap22q2/sample_info.csv')
            p_quantities = pd.read_csv(project_dir / 'data/raw/gygi_lab/protein_quant_current_normalized.csv')
            protein = p_quantities['Protein_Id'].str.strip('_HUMAN')
            p_quantities['protein'] = protein.str.split('|').str[2]
            p_quantities = p_quantities.drop(['Protein_Id','Description','Group_ID'], axis=1)
            # drop peptides
            p_quantities = p_quantities.drop(p_quantities.columns[3:45], axis=1)
            p_quantities = pd.melt(p_quantities, id_vars=['Uniprot', 'Gene_Symbol', 'Uniprot_Acc', 'protein'], var_name='Name_Tissue',
                        value_name='quantity')
            p_quantities.Name_Tissue = p_quantities.Name_Tissue.replace(to_replace ='_TenPx*', value = '', regex = True)
            p_quantities['Tissue'] = p_quantities.Name_Tissue.str.extract(r'((?<=_).*$)', expand=True)
            p_quantities['stripped_cell_line_name'] = p_quantities.Name_Tissue.str.extract(r'(.+?)_', expand=True)
            p_quantities = p_quantities.drop(['Uniprot', 'Name_Tissue'], axis=1)
            p_quantities = p_quantities.merge(sample_info[['stripped_cell_line_name', 'DepMap_ID']], how='inner', on='stripped_cell_line_name')
            sample_info = []
            p_quantities.to_pickle(project_dir / 'processed/p_quantities_l1000.pkl')
        else:
            pass

        print("Creating PPI graphs")
        oe = OrdinalEncoder()
        oe.fit(expression['gene'].unique().reshape(-1,1))
        oe_length = expression['gene'].unique().shape[0]
        oe_l = oe.transform(expression['gene'].unique().reshape(-1,1))
        gene_l = expression['gene'].unique().reshape(-1,1)
        self.save_oe(oe_l, gene_l)
        ai = oe.transform(ppi_links_cell.protein_1.values.reshape(-1,1))
        aj = oe.transform(ppi_links_cell.protein_2.values.reshape(-1,1))
        edge_list_simple = torch.tensor(np.concatenate((ai.T, aj.T), axis=0), dtype=torch.long)
        expression['gene_ordinal'] = oe.transform(expression['gene'].values.reshape(-1,1))
        node_name= oe.inverse_transform(np.array([i for i in range(oe_length)]).reshape(-1, 1))
        self.save_oe(np.array([i for i in range(oe_length)]).reshape(-1, 1), node_name)
        cells = expression['cellosaurus_accession'].unique()
        data_list = []
        idx = 0
        
        for cell in tqdm(cells, position=0, leave=True):
            try:
                cell_expression = expression.loc[expression['cellosaurus_accession'] == cell]
                x=cell_expression.sort_values(by=['gene_ordinal'])[['tpm','cnv','variant_type_oh']]
                x_ppi = torch.cat([torch.from_numpy(x['tpm'].values).unsqueeze(-1),
                                    torch.from_numpy(x['cnv'].values).unsqueeze(-1),
                                    torch.from_numpy(np.array([np.array(i) for i in x['variant_type_oh'].values])),
                                    ],
                                    dim=-1).to(torch.float)
                data = Data(x=x_ppi,
                        edge_index=edge_list_simple,
                        cell_name=cell)
                data_list.append(data)
                idx = idx +1
            except:
                print('Failed at index {}'.format(idx))
                idx = idx +1

        torch.save(self.collate(data_list), self.processed_paths[0])

if __name__ == '__main__':
    ppi = PPIGraphsL1000(root='/workspace')
    print(ppi.len())
    
"""