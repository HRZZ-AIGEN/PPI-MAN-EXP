# Drug response modeling with explanations
Multimodal attention model for prediction of drug response based on Graph Neural Networks and Transformers to utilize the Protein Protein Interactions (PPI) in cancer cell lines from the [STRING](https://string-db.org/cgi/download?sessionId=biPKFt1UqkvP&species_text=Homo+sapiens) DB. Multimodal Attention Network in drug response predictions can be seen as representation learning problem of cancer cell line expression data and molecule representation learning to predict absolute or relative IC50 value after fitting the curve to experiment measurements.  
<img src="figures/Info MAN.png"
     alt="MAN-png"
     style="float: left; margin-right: 10px;margin-bottom: 14px;" />
<br/>

This is an ongoing project of drug response pipeline that has been accepted at ISMB/ECCB 2021 - Representation learning in biology special session on account of the abstract and the poster. The preprint contains the further developed work. https://representation.learning.bio/schedule 

# Data
Using publicly available data from: [CTRP](https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/CTRPv2.0_2015_ctd2_ExpandedDataset.zip)<a id="ctrp"></a>, [GDSC](https://www.cancerrxgene.org/downloads/bulk_download)<a id="gdsc"></a>, [NCI](https://wiki.nci.nih.gov/display/NCIDTPdata/NCI-60+Growth+Inhibition+Data), [PubChem](https://pubchem.ncbi.nlm.nih.gov/)<a id="pubchem"></a>, [STRING](https://string-db.org/cgi/download?sessionId=biPKFt1UqkvP&species_text=Homo+sapiens)<a id="string"></a>, [DepMap](https://depmap.org/portal/download/)<a id="depmap"></a>. The data is located on the following link and should be unpackd in `project workdir/data/raw`. All the data is publicli available and can be accessed using `DataCreation.ipynb`<br/> 

The expression data of cancer cell lines were downloaded from the DepMap portal, version 22q2:
- CCLE_expressions.csv, [link](https://ndownloader.figshare.com/files/34989919)
- CCLE_gene_cn.csv, [link](https://ndownloader.figshare.com/files/34989937)
- CCLE_mutations.csv, [link](https://ndownloader.figshare.com/files/34989940)
- CTRPv2.0_2015_ctd2_ExpandedDataset.zip, [link](https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/CTRPv2.0_2015_ctd2_ExpandedDataset.zip)
- PRISM 19Q4 -- (secondary-screen-dose-response-curve-parameters.csv), [link](https://ndownloader.figshare.com/files/20237739)
- Gygi Lab, [link](https://gygi.hms.harvard.edu/data/ccle/Table_S2_Protein_Quant_Normalized.xlsx). This includes mass spectrometry-based quantification of proteins and peptides for 375 cell lines, actually for 375 CCL experiments, [link](https://www.cell.com/cell/fulltext/S0092-8674(19)31385-6)
- STRING DB, 9606.protein.info.v11.5.txt.gz and 9606.protein.links.full.v11.5.txt.gz, [link](https://string-db.org/cgi/download?sessionId=biPKFt1UqkvP&species_text=Homo+sapiens). These links among proteins and combined intensity of protein links were downloaded.
- PubChemDB, [link](https://pubchem.ncbi.nlm.nih.gov/). This is a database of chemical compounds with corresponding attributes, as well as generated features such as *inchi key, smiles, xlogp, fingerprint,* ...

# Model
We experiment with different types of models, to find the best parameters considering R squared, and MSE between labels and predictions. Preprocessing the [DepMap](#depmap) and [STRING](#string) data on one side and molecule graphs from [PubChem](#pubchem) portal on the other our experiments are conducted with two experiment settings: 
1. Take only the relevant pathways from [KEGG](https://www.genome.jp/kegg/)
2. Include all the pathways from the [STRING](#string)

To create the model Multimodal Attention Network we experiment with GAT[[1]](#1), GTN [[2]](#2) and MAT[[3]](#3). 

## Explanations

By analyzing attention weights and other importance factors we interpret interactions within the protein network. We demonstrate node interactions learned by the network color code them by signaling pathways and with focus on tumor specific genes (e.g. AUXIN, PTEN) for ovarian carcinoma pathways as in the example below.

<img src="figures/PPI pathway explanation.png"
     alt="MAN-png"
     style="float: left; margin-right: 10px;margin-bottom: 14px;" />
<br/>

And we interpret the results with self-explaining model indicating relevance of cancer cell line features to a specific compound regarding the response signal based on MEGAN [[4]](#4) explanations. 

# Installation
The repository is best installed via docker container located in `/start_docker_container_pytorch20` directory where the `dockerfile` and `requirements.txt` files. Also the directory `/.devcontainer` can be used to run `remote-containers.openFolder` in VSCode on remote ssh server. For more information check out [link](https://code.visualstudio.com/docs/devcontainers/containers)<br/>
Codes are working on newest version [pytorch](https://pytorch.org/) and [python](https://www.python.org/). After installation of the environment we recommend downloading data form web sources listed within `DataCreation.ipynb`. These are necessary steps of data acquisition, a prerequirenment to run the training below. <br/>

## Run training
After creation of all the necessary .pkl data files, you can run the training of benchmark and other experiments. The creation of 
Train regression model on DRP:<br/>
<pre><code>python src/train_drp.py --split=random --seed 42 --dataset NCI60DRP --batch_size 32 --epochs 2 --gpu 1 --self_att _self_att  </code></pre>

<pre><code>python src/train_drp.py --split=random --seed 42 --dataset NCI60DRPApr --batch_size 32 --epochs 2 --gpu 1 --self_att _self_att </code></pre>

# References
[1] <a id="1"></a>
Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. arXiv preprint arXiv:1710.10903.

[2] <a id="2"></a>Dwivedi, V. P., & Bresson, X. (2020). A generalization of transformer networks to graphs. arXiv preprint arXiv:2012.09699.

[3] <a id="3"></a>Maziarka, Ł., Danel, T., Mucha, S., Rataj, K., Tabor, J., & Jastrzębski, S. (2020). Molecule attention transformer. arXiv preprint arXiv:2002.08264.

[4] <a id="4"></a>Teufel, J., Torresi, L., Reiser, P., & Friederich, P. (2022). MEGAN: Multi-Explanation Graph Attention Network. arXiv preprint arXiv:2211.13236.