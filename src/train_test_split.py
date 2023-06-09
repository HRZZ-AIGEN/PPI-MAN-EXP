from pathlib import Path
import pandas as pd
import random
from create_graphs_mol import  MolecularGraphsGDSC, MolecularGraphs, MolecularGraphsCTRP
from create_graphs_ppi import PPIGraphsL1000, PPIGraphsDRP
import requests
import zipfile
import shutil
from rdkit import Chem, DataStructs
import numpy as np

def save_splits(save_dir, data):
    test = data.sample(random_state=42, frac=0.1)
    test.to_csv(save_dir / 'test.csv')
    ttest = data.sample(random_state=42, frac=0.02)
    ttest.to_csv(save_dir / 'test_testing.csv')
    data = data.drop(test.index)
    
    val = data.sample(random_state=42, frac=0.1)
    val.to_csv(save_dir / 'val.csv')
    tval = data.sample(random_state=42, frac=0.02)
    tval.to_csv(save_dir / 'val_testing.csv')
    
    train = data.drop(val.index)
    train.to_csv(save_dir / 'train.csv')
    # Saving 5% of data for overfitting with NN
    train_testing = train.sample(random_state=42, frac=0.06)
    train_testing.to_csv(save_dir / 'train_testing.csv')

def create_gdsc_split(split='random', self_att=''):
    """Creates a random split, used in benchmarks"""
    random.seed(42)
    root = Path(__file__).resolve().parents[1].absolute()

    # downloads data folder if it doesn't exist
    data_dir = (root / 'data')
    if not data_dir.exists():
        print("Downloading data (1.4 GB)")
        url = "https://www.dropbox.com/sh/h4tpjd64ebemo06/AADLEKi0844C0PQbk-X1ajk0a?dl=1"

        with requests.get(url, stream=True) as r:
            with open(root / "data.zip", 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print("Data downloaded!")
        data_dir.mkdir()
        with zipfile.ZipFile(root / "data.zip", "r") as zip_ref:
            zip_ref.extractall(data_dir)

    print("Creating train-val-test splits")
    if split == 'random' or split == 'paccmann':
        if split == 'random':
            save_dir = (root / 'data/processed/gdsc_benchmark')
        if split == 'paccmann':
            save_dir = (root / 'data/processed/paccmann')
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
            
        if Path(root / 'data/processed/gdsc_labeled_data.csv').exists():
            gdsc= pd.read_csv(root / 'data/processed/gdsc_labeled_data.csv')
            
        else:
            ppi_graphs = PPIGraphsL1000('/workspace', self_att=self_att) # load and/or create PPI graphs
            gdsc = pd.read_pickle(root / 'data/raw/GDSC2/GDSC2_IC50_PUBCHEM.pkl')

            if split == 'paccmann':
                gdsc['LN_IC50'] = (gdsc['LN_IC50']-gdsc['LN_IC50'].min())/(gdsc['LN_IC50'].max()-gdsc['LN_IC50'].min())
            sample_info = pd.read_csv(root / 'data/raw/depmap22q2/sample_info.csv')[['COSMICID', 'RRID']]

            cell_names = []
            for i in ppi_graphs:
                cell_names.append(i.cell_name)

            gdsc = gdsc.merge(sample_info, how='inner', left_on='COSMIC_ID', right_on='COSMICID')
            gdsc = gdsc.loc[gdsc['RRID'].isin(cell_names)]
            gdsc.drop_duplicates(subset=['drug_name', 'RRID'], inplace=True)
            gdsc = gdsc.loc[~gdsc['smiles'].str.contains('\.')]
            # gdsc.rename(columns={
            #                     # 'Drug name': 'drug_name',
            #                     #  'CanonicalSMILES': 'smiles',
            #                      'LN_IC50': 'pIC50'}, inplace=True)
            gdsc.to_csv(root / 'data/processed/gdsc_labeled_data.csv')
            print(gdsc.head())
            # save train-test-splits
        save_splits(save_dir, gdsc)
            
    if split == 'blind':
        save_dir = Path(root / 'data/processed/gdsc_benchmark_blind')
        if not Path(save_dir / 'val.csv').exists():
            save_dir.mkdir(parents=True, exist_ok=True)
            ppi_graphs = PPIGraphsL1000('/workspace', self_att=self_att)  # load and/or create PPI graphs
            cell_names = []
            for i in ppi_graphs:
                cell_names.append(i.cell_name)
            gdsc = get_gdsc_data(root, cell_names)
            
            if not Path(root / 'data/processed/gdsc_labeled_data.csv').exists():
                gdsc.to_csv(root / 'data/processed/gdsc_labeled_data.csv')

            # save train-test-splits
            """Make 3 test set settings concatenated in one:
                1. Blind cells
                2. Blind drugs
                3. Blind cells and drugs"""
            # blind drugs
            drugs = gdsc['drug_name'].unique()
            n_drugs = int(len(drugs) * 0.15)  # number of unique drugs to draw
            unique_drugs = random.sample(list(drugs), n_drugs)
            blind_drugs = gdsc.loc[gdsc['drug_name'].isin(unique_drugs)]
            data = gdsc.drop(blind_drugs.index)

            # double blind
            double_blind_cells = blind_drugs['RRID'].unique()
            n_cells = int(len(double_blind_cells) * 0.1)
            double_blind_cells = random.sample(list(double_blind_cells), n_cells)
            double_blind = blind_drugs.loc[blind_drugs['RRID'].isin(double_blind_cells)]
            data = data.loc[~data['RRID'].isin(double_blind['RRID'].unique())]

            # blind cells
            blind_cells = data['RRID'].unique()
            n_cells = int(len(blind_cells) * 0.1)
            blind_cells = random.sample(list(blind_cells), n_cells)
            blind_cells = data.loc[data['RRID'].isin(blind_cells)]
            data = data.drop(blind_cells.index)
            val = data.sample(random_state=42, frac=0.1)
            data = data.drop(val.index)

            blind_drugs.to_csv(save_dir / 'blind_drugs.csv')
            blind_cells.to_csv(save_dir / 'blind_cells.csv')
            double_blind.to_csv(save_dir / 'double_blind.csv')
            pd.concat([blind_drugs, blind_cells, double_blind]).to_csv(save_dir / 'test.csv')
            val.to_csv(save_dir / 'val.csv')
            data.to_csv(save_dir / 'train.csv')

    MolecularGraphsGDSC(root) # if !exists -> create
    return save_dir

def get_gdsc_data(root, cell_names):
    data = pd.read_pickle(root / 'data/raw/GDSC2/GDSC2_IC50_PUBCHEM.pkl')
    sample_info = pd.read_csv(root / 'data/raw/depmap22q2/sample_info.csv')[['COSMICID', 'RRID']]
    # data = pd.read_pickle(root / 'raw/GDSC2/GDSC2_IC50_PUBCHEM.pkl')
    sample_info = pd.read_csv(root / 'data/raw/depmap22q2/sample_info.csv')[['COSMICID', 'RRID']]
    data = data.merge(sample_info, how='inner', left_on='COSMIC_ID', right_on='COSMICID')
    data = data.loc[data['RRID'].isin(cell_names)]
    data.drop_duplicates(subset=['drug_name', 'RRID'], inplace=True)
    data = data.loc[~data['smiles'].str.contains('\.')]
    # data.rename(columns={
    #                     # 'Drug name': 'drug_name',
    #                     #  'CanonicalSMILES': 'smiles',
    #                         'LN_IC50': 'pIC50'}, inplace=True)
    data.reset_index(inplace=True)
    return data

def get_nci_data(root, cell_names):
    data = pd.read_pickle('/workspace/data/raw/NCI60June2022/NCI60_IC50_PUBCHEM_PUBLIC.pkl')
    data = data.loc[data['RRID'].isin(cell_names)]
    data.drop_duplicates(subset=['pubchem_id', 'RRID'], inplace=True)
    data.reset_index(inplace=True)
    return data 

def train_test_split(dataset='GDSC', split='random', 
                     self_att='', new_data = ''):
    """Creates different splits, on different datasets from PharmacoGX R package"""
    random.seed(42)
    root = Path(__file__).resolve().parents[1].absolute()

    data_dir = (root / 'data')
    if not data_dir.exists():
        print("Downloading data (1.4 GB)")
        url = "https://www.dropbox.com/sh/h4tpjd64ebemo06/AADLEKi0844C0PQbk-X1ajk0a?dl=1"

        with requests.get(url, stream=True) as r:
            with open(root / "data.zip", 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print("Data downloaded!")

        data_dir.mkdir()
        with zipfile.ZipFile(root / "data.zip", "r") as zip_ref:
            zip_ref.extractall(data_dir)

    print("Creating train-val-test splits")
    save_dir = (root / 'data/processed/{}_{}'.format(dataset + new_data, 
                                                     split+self_att))
    train_file = (save_dir / 'train.csv')
    if not train_file.exists():
        print('Making {} splits! \n '.format(dataset + new_data))
        save_dir.mkdir(parents=True, exist_ok=True)
        ppi_graphs = PPIGraphsL1000('/workspace', self_att=self_att)  # if !exists -> create
        cell_names = []
        for i in ppi_graphs:
            cell_names.append(i.cell_name)
        if dataset == 'GDSC':
            data = get_gdsc_data(root, cell_names)
        elif dataset == 'NCI60' or dataset == 'NCI':
            data = get_nci_data(root, cell_names)
        elif dataset == 'CTRP':
            data = pd.read_pickle('/workspace/data/raw/CTRP/CTRP_all_v2.pkl')
            data = data.loc[data['RRID'].isin(cell_names)]
            data.drop_duplicates(subset=['pubchem_id', 'RRID'], inplace=True)
            data.reset_index(inplace=True)
            print('lenght of data: ', len(data))
            
            # TODO create train_test_splits for DRP:
        elif dataset == 'NCI60DRP' or dataset == 'NCI':
            ppi_graphs = PPIGraphsDRP('/workspace', self_att=self_att, 
                                      new_data = new_data)  # if !exists -> create
            cell_names = []
            for i in ppi_graphs:
                cell_names.append(i.cell_name)
            data = get_nci_data(root, cell_names)
            
        # MolecularGraphs(root / 'data/processed')  # if !exists -> create

        # bad_cids = [117560, 23939, 0, 23976]
        # drop Cr atom from CTRP
        # data = data.drop(data.loc[data["pubchem_id"].isin(bad_cids)].index)

        if split == 'random':
            """Randomly split on unknown drugs"""
            save_splits(save_dir, data)

        if split == 'blind':
            """Make 3 test set settings concatenated in one:
                1. Blind cells
                2. Blind drugs
                3. Blind cells and drugs"""
            # blind drugs
            drugs = data['pubchem_id'].unique()
            n_drugs = int(len(drugs) * 0.15)  # number of unique drugs to draw
            unique_drugs = random.sample(list(drugs), n_drugs)
            blind_drugs = data.loc[data['pubchem_id'].isin(unique_drugs)]
            data = data.drop(blind_drugs.index)

            # double blind
            double_blind_cells = blind_drugs['RRID'].unique()
            n_cells = int(len(double_blind_cells) * 0.1)
            if dataset == 'NCI60':
                n_cells = 5
                
            double_blind_cells = random.sample(list(double_blind_cells), n_cells)
            double_blind = blind_drugs.loc[blind_drugs['RRID'].isin(double_blind_cells)]
            data = data.loc[~data['RRID'].isin(double_blind['RRID'].unique())]

            # blind cells
            blind_cells = data['RRID'].unique()
            n_cells = int(len(blind_cells) * 0.1)
            if dataset == 'NCI60':
                n_cells = 5
                
            blind_cells = random.sample(list(blind_cells), n_cells)
            blind_cells = data.loc[data['RRID'].isin(blind_cells)]
            data = data.drop(blind_cells.index)
            val = data.sample(random_state=42, frac=0.1)
            data = data.drop(val.index)

            blind_drugs.to_csv(save_dir / 'blind_drugs.csv')
            blind_cells.to_csv(save_dir / 'blind_cells.csv')
            double_blind.to_csv(save_dir / 'double_blind.csv')
            pd.concat([blind_drugs, blind_cells, double_blind]).to_csv(save_dir / 'test.csv')
            val.to_csv(save_dir / 'val.csv')
            data.to_csv(save_dir / 'train.csv')
            train_testing = data.sample(random_state=42, frac=0.2)
            train_testing.to_csv(save_dir / 'train_testing.csv')

        if split.startswith('drug'):
            # work with 2 datasets so reload data
            data = pd.read_pickle(root / 'data/processed/labeled_data.pkl', index_col=0)
            data = data.loc[data['RRID'].isin(cell_names)]
            data = data.loc[data['dataset_name'] != 'NCI60']

            bad_cids = [117560, 23939, 0, 23976]
            # drop Cr atom from CTRP
            data = data.drop(data.loc[data["pubchem_id"].isin(bad_cids)].index)

            # Sample 125 drugs from GDSC dataset and calculate similarity from CTRP to these drugs
            other_drugs = random.sample(list(data.loc[data['dataset_name'] == 'GDSC']['pubchem_id'].unique()), 125)

            if dataset == 'GDSC':
                other = data.loc[data['dataset_name'] == 'CTRP']
                data = data.loc[data['pubchem_id'].isin(other_drugs)]
            else:
                data = data.loc[data['dataset_name'] == 'CTRP']
                other = data.loc[data['pubchem_id'].isin(other_drugs)]

            # drop overlapping drugs only on dissimilarity split
            if split == 'drug_dissimilarity':
                overlap_drugs = data.merge(other, how='inner', on='pubchem_id')
                overlap_drugs = overlap_drugs['pubchem_id'].unique()

                other = other.loc[~other['pubchem_id'].isin(overlap_drugs)]
                data = data.loc[~data['pubchem_id'].isin(overlap_drugs)]

                # Firstly filter on scaffolds
                data = data.loc[~data['scaffolds'].isin(other['scaffolds'])]

            # Find 125 most (dis)similar drugs from those in the gdsc dataset based on Tanimoto similarity
            other_mols = [Chem.MolFromSmiles(x) for x in list(other['smiles'].unique())]
            other_fps = [Chem.RDKFingerprint(x) for x in other_mols]

            data_cid = [x for x in list(data['pubchem_id'].unique())]
            data_mols = [Chem.MolFromSmiles(x) for x in list(data['smiles'].unique())]
            data_fps = [Chem.RDKFingerprint(x) for x in data_mols]

            average_similarity_data = []
            for i in range(len(data_fps)):
                similarity_of_mol = []
                for j in range(len(other_fps)):
                    similarity_of_mol.append(DataStructs.FingerprintSimilarity(data_fps[i], other_fps[j]))
                average_similarity_data.append(np.mean(similarity_of_mol))

            if split == 'drug_dissimilarity':
                data_average_similarity = pd.DataFrame(
                    {'pubchem_id': data_cid,
                     'average_similarity_ctrp': average_similarity_data}
                ).sort_values(by='average_similarity_ctrp', ascending=True)[:125]['pubchem_id']
                # returns most dissimilar drugs ascending = True

            else:
                data_average_similarity = pd.DataFrame(
                    {'pubchem_id': data_cid,
                     'average_similarity_ctrp': average_similarity_data}
                ).sort_values(by='average_similarity_ctrp', ascending=False)[:125]['pubchem_id']
                # returns most 125 similar drugs ascending=False

            data = data.loc[data['pubchem_id'].isin(data_average_similarity)]
            save_splits(save_dir, data)

        if split.startswith('cell'):
            # work with 2 datasets so reload data
            data = pd.read_pickle(root / 'data/processed/labeled_data.pkl', index_col=0)
            data = data.loc[data['RRID'].isin(cell_names)]
            data = data.loc[data['dataset_name'] != 'NCI60']

            bad_cids = [117560, 23939, 0, 23976]
            # drop Cr atom from CTRP
            data = data.drop(data.loc[data["pubchem_id"].isin(bad_cids)].index)

            other = data.loc[data['dataset_name'] != dataset]
            data = data.loc[data['dataset_name'] == dataset]

            overlap_cells = data.merge(other, how='inner', on='RRID')
            overlap_cells = overlap_cells['RRID'].unique()

            # find overlapping cell lines
            if split == 'cell_dissimilarity':  # drop only overlapping cells
                data = data.loc[~data['RRID'].isin(overlap_cells)]

            if split == 'cell_similarity':  # keep only overlapping cells
                data = data.loc[data['RRID'].isin(overlap_cells)]

            save_splits(save_dir, data)

    return save_dir

if __name__ == "__main__":
    train_test_split(dataset='NCI60DRP', split='random', 
                     self_att='', new_data='Apr')