import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.pair_graphs import PairDataset, collate_batch, InferenceDataset
from src.create_graphs_ppi import PPIGraphsL1000, PPIGraphsDRP
import numpy as np
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from src.train_classification_models import MultimodalAttentionNetClassification
from sklearn.metrics import average_precision_score, roc_auc_score
from src.train_drp import MultimodalAttentionNet, Conf

conf = Conf(
    lr=1e-3,
    batch_size=32,
    epochs=300,
    reduce_lr=True
).to_hparams()

class Evaluation:
    def __init__(self, root_dir, dataset = 'GDSC', split = 'random', 
                 ppi_depth = 3, seed = 42, gpu = 3, mat_heads = 8, mat_depth = 8,
                 ckpt_path='', model = None):
        
        self.dataset = dataset
        self.root_dir = Path(root_dir)
        
        self.cells = []
        self.ckpt_path = ckpt_path
        self.model = model
        if gpu in [0,1,2,3]:
            device = 'cuda:{}'.format(gpu)
        else: 
            device= 'cpu'
        self.device = torch.device(device)
        if ppi_depth == 3:
            self.model_path = Path(self.root_dir / Path('models/{}_{}_{}/'.format(
                dataset, split, seed)))
            if self.ckpt_path == '':
                self.ckpt_path = str(list(self.model_path.glob('**/*.ckpt'))[0])
            self.model = self.model.load_from_checkpoint(
                self.ckpt_path, mat_heads = mat_heads, mat_depth = mat_depth,
                data_dir = None)
            
            self.model.to(self.device)
        if dataset == 'NCI60DRP':
            self.ppi = PPIGraphsDRP(self.root_dir)
            for i in self.ppi:
                self.cells.append(i.cell_name)
        else: 
            for i in PPIGraphsL1000(self.root_dir):
                self.cells.append(i.cell_name)

        if ppi_depth != 3:
            self.model_path = Path(self.root_dir / Path('models/{}_{}_{}_ppi_depth_{}/'.format(
                dataset, split, seed, ppi_depth)))
            
            self.ckpt_path = str(list(self.model_path.glob('**/*.ckpt'))[0])
            self.model = MultimodalAttentionNetClassification(conf, ppi_depth=ppi_depth,
                        data_dir=None).load_from_checkpoint(self.ckpt_path, data_dir=None, 
                            ppi_depth=ppi_depth, map_location = self.device)

        self.data_path = Path(root_dir / Path('data/processed/{}_{}/'.format(dataset, split)))
        self.model.eval()

    def top_k_interactions(self, k_interactions=25):
        # interactions in NCI cells
        nci_cells = list(self.labels.loc[self.labels['dataset'] == 'NCI60']['cellosaurus_accession'].unique())
        gdsc_data = self.labels.loc[
            (self.labels['dataset'] == 'GDSC') & (self.labels['dataset'] == 'CTRP')
            ]
        nci_cells = list(gdsc_data.loc[gdsc_data['cellosaurus_accession'].isin(nci_cells)].unique())

        for cell in nci_cells:
            try:
                _, links = self.ppi_graph(cell)
                links.sort_values(by='attention', descending=True, inplace=True)
                top_k = links.iloc[:k_interactions]

            except:
                print('Cell {} not in DepMap'.format(cell))

    def interactions_dict(self, top_k_interactions):
        # use top_k for generating top k links
        def top_k(attention_dict, cell_name, top_k_interactions, mean_attention):
            attention = attention_dict[cell_name]
            attention['difference'] = abs(attention['attention'] - mean_attention)
            attention = attention.sort_values(by='difference', ascending=False).iloc[:top_k_interactions]
            attention['path'] = attention['protein_1'] + '-' + attention['protein_2']
            return list(attention['path'])

        attention, cellosaurus = self.attention_links()  # generate attention for cell lines

        attentions = []
        for i in attention:
            attentions.append(i['attention'].values)

        mean_attention = np.array(attentions).sum(axis=0) / 6
        attention_dict = dict(zip(cellosaurus, attention))
        interactions_list = []
        cellosaurus_accession = []

        for cell in self.cells:
            cell_interactions = top_k(attention_dict, cell, top_k_interactions, mean_attention)
            interactions_list.append(cell_interactions)
            cellosaurus_accession.append(cell)

        interactions_dict = dict(zip(cellosaurus_accession, interactions_list))

        return interactions_dict

    def load_links(self):
        if self.dataset == 'NCI60DRP':
            return pd.read_pickle(self.root_dir / 'data/processed/NCI60DRP_random/ppi_links_drp.pkl')
        else:
            return pd.read_pickle(self.root_dir / 'data/processed/ppi_links_l1000.pkl')
            
            
    def predict(self, drug_id, smiles, 
                cellosaurus_accession,
                batch_size=1):
        query = InferenceDataset(drug_id=drug_id, smiles=smiles,
                                 cellosaurus_accession=cellosaurus_accession, 
                                 dataset=self.dataset)

        test_loader = DataLoader(query, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                                 collate_fn=collate_batch)
        
        links = self.load_links()
    
        for batch in test_loader:
            adj_mat, dist_mat, x = batch[0]
            adj_mat, dist_mat, x = adj_mat.to(self.device), dist_mat.to(self.device), x.to(self.device)
            x_ppi = batch[1].x
            x_ppi = x_ppi.to(self.device)
            ppi_edge_index = batch[1].edge_index
            ppi_edge_index = ppi_edge_index.to(self.device)
            ppi_batch = batch[1].batch
            ppi_batch = ppi_batch.to(self.device)
            mask = torch.sum(torch.abs(x), dim=-1) != 0
            mask = mask.to(self.device)
            y_hat, attention = self.model(x,
                adj_mat,
                dist_mat,
                mask,
                x_ppi,
                ppi_edge_index,
                ppi_batch, 
                att_weights=True)

        y_hat = y_hat.squeeze(-1).detach().cpu().numpy()
        # print(attention[0].shape, attention[1].shape)
        att_edge_index = attention[0].detach().cpu().numpy()
        
        heads = attention[1].shape[1]
        attention = torch.sum(attention[1], dim=1).detach().cpu().numpy() / heads 
        # links['attention'] = attention
        return y_hat, attention, att_edge_index
    
    def attention_links(self):
        cells = self.cells
        df = pd.DataFrame([])
        for cell in tqdm(cells):
            #try:
            _, links, att_edge_index = self.predict(drug_id=['benzene'],
                                    smiles=["C1=CC=CC=C1"],
                                    cellosaurus_accession=[cell], 
                                    batch_size=1)

            oe = np.load(self.root_dir / 'data/processed/pytorch_graphs/PPIDRP/oe{}.npy'.format(cell), allow_pickle=True)
            # Make a dataframe with columns: attention, 
            # ppi_edges, cellosaurus_accession, ppi.get_node_name
            oe.sort()
            
            df_t = pd.DataFrame({'attention': links, 
                               'protein_1': att_edge_index[0],
                               'protein_2': att_edge_index[1], 
                               'cell': [cell for i in range(len(links))],
                               'p1_name': [oe[1][i] 
                                           for i in att_edge_index[0]],
                               'p2_name': [oe[1][i]
                                           for i in att_edge_index[1]]})
            
            df = pd.concat([df, df_t])
        return df

    def eval_test_sets(self, split='random', how='average'):
        id_blind_cells = []
        ap_blind_cells = []
        auc_blind_cells = []

        id_double_blind = []
        ap_double_blind = []
        auc_double_blind = []

        id_blind_drugs = []
        ap_blind_drugs = []
        auc_blind_drugs = []

        test_sets = list(self.data_path.glob('**/*')) #list all file is data_path directory

        try:
            blind_cells = [test for test in test_sets if str(test).endswith('blind_cells.csv')][0]
            blind_drugs = [test for test in test_sets if str(test).endswith('blind_drugs.csv')][0]
            double_blind = [test for test in test_sets if str(test).endswith('double_blind.csv')][0]

            if how == 'average':
                blind_cells = pd.read_csv(blind_cells, index_col=0)
                ap, auc = self.calculate_metrics(blind_cells, split)
                ap_blind_cells.append(ap)
                auc_blind_cells.append(auc)
                id_blind_cells.append('average')

                blind_drugs = pd.read_csv(blind_drugs, index_col=0)
                ap, auc = self.calculate_metrics(blind_drugs, split)
                id_blind_drugs.append('average')
                ap_blind_drugs.append(ap)
                auc_blind_drugs.append(auc)

                double_blind = pd.read_csv(double_blind, index_col=0)
                ap, auc = self.calculate_metrics(double_blind, split)
                id_double_blind.append('average')
                ap_double_blind.append(ap)
                auc_blind_drugs.append(auc)

            else:
                # calculate blind drugs
                blind_cells = pd.read_csv(blind_cells, index_col=0)
                cell_ids = blind_cells['cellosaurus_accession'].unique()
                for i in cell_ids:
                    df = blind_cells.loc[blind_cells['cellosaurus_accession'] == i]
                    ap, auc = self.calculate_metrics(df, split)
                    id_blind_cells.append(i)
                    ap_blind_cells.append(ap)
                    auc_blind_cells.append(auc)
                print('Finished evaluating blind cells setting')

                # calculate blind cells
                blind_drugs = pd.read_csv(blind_drugs, index_col=0)
                blind_drugs_ids = blind_drugs['pubchem_cid'].unique()
                for i in blind_drugs_ids:
                    df = blind_drugs.loc[blind_drugs['pubchem_cid'] == i]
                    ap, auc = self.calculate_metrics(df, split)
                    id_blind_drugs.append(i)
                    ap_blind_drugs.append(ap)
                    auc_blind_drugs.append(auc)
                print('Finished evaluating blind drugs setting')

                double_blind = pd.read_csv(double_blind, index_col=0)
                double_blind_cell_ids = double_blind['cellosaurus_accession'].unique()
                for i in double_blind_cell_ids:
                    df = double_blind.loc[double_blind['cellosaurus_accession'] == i]
                    ap, auc = (self.calculate_metrics(df, split))
                    id_double_blind.append(i)
                    ap_double_blind.append(ap)
                    auc_double_blind.append(auc)
                print('Finished evaluating double blind setting')

            results_dict = {'blind_cells': [id_blind_cells, ap_blind_cells, auc_blind_cells],
                            'blind_drugs': [id_blind_drugs, ap_blind_drugs, auc_blind_drugs],
                            'double_blind': [id_double_blind, ap_double_blind, auc_double_blind]}

            return results_dict

        except: #When there are no individual splits
            test_set = [test for test in test_sets if str(test).endswith('test.csv')][0]
            test_df = pd.read_csv(test_set, index_col=0)
            ap, auc = self.calculate_metrics(test_df, split)

            results_dict = {'AP': ap,
                            'AUC': auc}

            return results_dict

    def calculate_metrics(self, data, split='random'):
        """Calculate metrics on CPU if there are many elements to prevent memory errors on the GPu"""
        try:
            dataset = PairDataset(self.data_path / data)
            test_data = pd.read_csv(self.data_path / data)
        except:
            dataset = PairDataset(data)
            test_data = data

        test = DataLoader(dataset, 18, shuffle=False, 
                          num_workers=8, pin_memory=False, 
                          collate_fn=collate_batch)
        predictions = []

        for batch in test:
            adj_mat, dist_mat, x = batch[0]
            x_ppi = batch[1].x
            ppi_edge_index = batch[1].edge_index
            ppi_batch = batch[1].batch
            mask = torch.sum(torch.abs(x), dim=-1) != 0
            y_hat, _ = self.model(
                x,
                adj_mat,
                dist_mat,
                mask,
                x_ppi,
                ppi_edge_index,
                ppi_batch,
            )
            y_hat = y_hat.squeeze(-1).detach().cpu().numpy().flatten().tolist()
            predictions.append(y_hat)

        predictions = [item for sublist in predictions for item in sublist]
        prediction_df = pd.DataFrame({'predictions': predictions,
                                      'pubchem_cid': test_data['pubchem_cid'],
                                      'cellosaurus_accession': test_data['cellosaurus_accession'],
                                      'sensitivity_uM': test_data['sensitivity_uM'],
                                      'scaffolds': test_data['scaffolds']})
        train_data = pd.read_csv(self.data_path / 'train.csv', index_col=0)
        val_data = pd.read_csv(self.data_path / 'val.csv', index_col=0)
        if split == 'scaffold':
            prediction_df = prediction_df.loc[~prediction_df['scaffolds'].isin(train_data['scaffolds'])]
            prediction_df = prediction_df.loc[~prediction_df['scaffolds'].isin(val_data['scaffolds'])]

        try:
            ap = average_precision_score(prediction_df['sensitivity_uM'], prediction_df['predictions'])
            roc_auc = roc_auc_score(prediction_df['sensitivity_uM'], prediction_df['predictions'])
        except:
            ap = 0
            roc_auc = 0
        print('Calculated AP and AU-ROC for a single iteration, AP: {}, AUC: {}'.format(ap, roc_auc))
        return ap, roc_auc

def overlaps(dict1, dict2):
    overlap = []
    cells = []
    for key in dict1:
        list1 = dict1[key]
        list2 = dict2[key]
        overlap.append(len([value for value in list1 if value in list2]))
        cells.append(key)
    return overlap

def overlap_3(dict1, dict2, dict3):
    overlap = []
    cells = []
    for key in dict1:
        list1 = dict1[key]
        list2 = dict2[key]
        list3 = dict3[key]
        ol1 = [value for value in list1 if value in list2]
        overlap.append(len([value for value in ol1 if value in list3]))
        cells.append(key)
    return overlap

def disease_dicts(sample_info, dict1, disease_class):
    new_dict = dict1
    sample_info = sample_info.loc[sample_info['RRID'].isin(list(new_dict.keys()))]
    primary_disease_list = sample_info[disease_class].unique()
    disease_list = []
    interactions_list = []
    overlap = []
    for disease in primary_disease_list:
        disease_specific_ccl = sample_info.loc[sample_info[disease_class] == disease]['RRID'].unique()
        num_cells = len(disease_specific_ccl)
        disease_interactions = []
        for ccl in disease_specific_ccl:
            new_dict[ccl] = new_dict[ccl]
            disease_interactions.append(new_dict[ccl])

        disease_interactions = [item for sublist in disease_interactions for item in sublist]
        n_interactions = [0] * len(disease_interactions)
        interactions_dict = dict(zip(disease_interactions, n_interactions))

        # calculate num interactions
        for ccl in disease_specific_ccl:
            interactions = new_dict[ccl]
            for i in interactions:
                if i in interactions_dict.keys():
                    interactions_dict[i] += 1

        interactions_list.append(interactions_dict)
        disease_list.append(disease)

        return dict(zip(disease_list, interactions_list))

def disease_subtype_dict(sample_info, dict1):
    new_dict = dict1
    sample_info = sample_info.loc[sample_info['RRID'].isin(list(new_dict.keys()))]
    disease_subtype_list = []
    interactions_list = []

    for disease in sample_info['primary_disease'].unique():
        subtypes = sample_info.loc[sample_info['primary_disease'] == disease]['Subtype'].unique()
        for subtype in subtypes:
            cells = sample_info.loc[(sample_info['primary_disease'] == disease) &
                                    (sample_info['Subtype'] == subtype)]['RRID'].unique()
            num_cells = len(cells)
            disease_interactions = []
            for cell in cells:
                disease_interactions.append(new_dict[cell])

            disease_interactions = [item for sublist in disease_interactions for item in sublist]
            n_interactions = [0] * len(disease_interactions)
            interactions_dict = dict(zip(disease_interactions, n_interactions))

            for cell in cells:
                interactions = new_dict[cell]
                for i in interactions:
                    if i in interactions_dict.keys():
                        interactions_dict[i] += 1

            interactions_list.append(interactions_dict)
            disease_subtype_list.append(str('{}_{}'.format(disease, subtype)))

    return dict(zip(disease_subtype_list, interactions_list))

if __name__ == '__main__':
    conf = Conf(
    lr = 1e-3,
    batch_size = 32,
    epochs = 300,
    reduce_lr = True,
    ppi_depth = 3,
    mat_depth = 4,
    mat_heads = 4,).to_hparams()

    model = MultimodalAttentionNet(conf, data_dir=None, 
    mat_depth=4, mat_heads=4, ppi_depth=3)
    drp = Evaluation('/workspace', dataset='NCI60DRP', 
               split='random', ppi_depth=3, 
               seed=42, mat_depth=4, mat_heads=4, 
                ckpt_path='models/NCI60DRP_random_42/'+
                '1680458624/checkpoint/epoch=212-step=1957470.ckpt', 
                model = model)
    
    attention = drp.attention_links()
    