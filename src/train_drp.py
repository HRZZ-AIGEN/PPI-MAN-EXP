import dataclasses
import shutil
from abc import ABC
from pathlib import Path
from pprint import pformat
from time import time
from typing import Dict, Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from multimodal_attention_network import MMAtt
from torchmetrics import R2Score, SpearmanCorrCoef
from torchmetrics.functional import mean_squared_error
from torch.optim import Adam
from torch.utils.data import DataLoader
from pair_graphs import (collate_batch, 
                         PairDatasetBenchmark, PairDataset, 
                         collate_wrapper)
from create_graphs_mol import MolData
from train_test_split import train_test_split
import click
import numpy as np
import gc
import os

root = Path(__file__).resolve().parents[1].absolute()
VERSION = str(int(time()))
@dataclasses.dataclass(frozen=True)
class Conf:
    gpus: int = 2
    seed: int = 42
    use_16bit: bool = False
    save_dir = '{}/models/'.format(root)
    lr: float = 1e-4
    batch_size: int = 64
    epochs: int = 300
    ckpt_path: Optional[str] = None
    reduce_lr: Optional[bool] = False
    problem: Optional[str] = 'regression'
    self_att: Optional[str] = ''
    testing: str = ''
    ppi_depth: int = 3
    mat_depth: int = 8
    mat_heads: int = 8
    dim_in_gat: int = 5
    new_data: str = ''
    test_5000: bool = False
    profiler: Optional[str] = None
    version: int = 0
    num_workers: int = 4
    gat_model_dim:int = 64

    def to_hparams(self) -> Dict:
        excludes = [
            'ckpt_path',
            'reduce_lr',
        ]
        return {
            k: v
            for k, v in dataclasses.asdict(self).items()
            if k not in excludes
        }

    def __str__(self):
        return pformat(dataclasses.asdict(self))


class MultimodalAttentionNet(pl.LightningModule, ABC):
    def __init__(
            self,
            hparams,
            data_dir: Optional[Path] = None,
            ppi_depth=3,
            reduce_lr: Optional[bool] = True,
            dim_in_gat: int = 5,
            add_self_loops: Optional[bool] = True,
            mat_depth: int = 8,
            mat_heads: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        # self.hparams = hparams
        self.reduce_lr = reduce_lr
        self.data_dir = data_dir
        
        self.model = MMAtt(ppi_depth = ppi_depth, 
                           dim_in_gat = dim_in_gat, 
                           add_self_loops = add_self_loops,
                           mat_depth = mat_depth, 
                           mat_heads = mat_heads)
        
        self.RMSE = mean_squared_error
        self.r2 = R2Score()
        self.loss_fn = torch.nn.MSELoss()
        self.spearman = SpearmanCorrCoef()
        self.mean_squared_error = torch.nn.MSELoss()

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        pl.seed_everything(hparams['seed'])
        # print(self.hparams)

    def forward(self, x, adj_mat, dist_mat, mask, 
                x_ppi, ppi_edge_index, ppi_batch, att_weights=False):
        
        out, attention_ppi = self.model(x, adj_mat, dist_mat, mask, 
                                        x_ppi, ppi_edge_index, ppi_batch, 
                                        att_weights=att_weights)
        
        return out, attention_ppi

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, training=True)
        self.log("train_step/lr", self.trainer.optimizers[0].param_groups[0]['lr'], 
                 batch_size=self.hparams.batch_size)
        self.log("train_step/loss", loss.detach(), batch_size=self.hparams.batch_size)
        return loss
    
    def on_train_epoch_end(self) -> None:
        loss = torch.stack([x.get('loss') for x in 
                            self.train_step_outputs]).mean()
        rmse_avg = torch.stack([x.get('train_rmse_step') for x in 
                                self.train_step_outputs]).mean()
        #r2_avg = np.stack([x.get('train_r2_step') for x in 
        #                        self.train_step_outputs]).mean()
        self.log('train_epoch/loss', loss, on_epoch=True, prog_bar=True,
                 batch_size=self.hparams.batch_size, sync_dist=True)
        self.log('train_epoch/RMSE', rmse_avg, on_epoch=True, prog_bar=True,
                 batch_size=self.hparams.batch_size, sync_dist=True)
        self.log('train_epoch/R2', self.r2.compute(), on_epoch=True, prog_bar=True,
                 batch_size=self.hparams.batch_size, sync_dist=True)
        self.train_step_outputs.clear()
        del loss, rmse_avg #, r2_avg
        return 
    
    def validation_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx, training=False)
        self.log('val_step/loss', metrics.get("loss"),
                 prog_bar=False, on_step=True, add_dataloader_idx=True,
                 batch_size=self.hparams.batch_size, sync_dist=True)
        
        rmse = self.RMSE(metrics.get("predictions"), metrics.get("targets"), squared=False) 
        self.log('val_step/RMSE', rmse,
                 prog_bar=False, on_step=True, add_dataloader_idx=True,
                 batch_size=self.hparams.batch_size, sync_dist=True)
        
        metrics['val_rmse_step'] = rmse 
        self.validation_step_outputs.append(metrics)
        """       
        if batch_idx == 0:
            print(pformat(metrics))"""
        return metrics

    def detach_delete(self, x):
        x = x.detach()
        del x
        return None
    
    def on_validation_epoch_end(self):
        # print(pformat(outputs))        
        
        loss = torch.stack([x.get('loss') for x in 
                            self.validation_step_outputs]).mean()
        #for x in self.validation_step_outputs:
        #    print(x.get('val_rmse_step'))
        rmse_avg = torch.stack([x.get('val_rmse_step') for x in 
                                self.validation_step_outputs]).mean()

        predictions = torch.cat([x.get('predictions') for x in 
                                 self.validation_step_outputs], 0)
        targets = torch.cat([x.get('targets') for x in 
                             self.validation_step_outputs], 0)
        rmse = self.RMSE(predictions, targets,squared=False) 
        r2 = self.r2(predictions, targets)
        # predictions = predictions.type(torch.FloatTensor)
        spearman = self.spearman(predictions, targets)
        
        self.log('val_epoch/RMSE', rmse, prog_bar=True, 
                 on_epoch=True, add_dataloader_idx=True,
                 batch_size=self.hparams.batch_size, sync_dist=True)
        
        self.log('val/RMSE_avg', rmse_avg, prog_bar=True, 
                 on_epoch=True, add_dataloader_idx=True,
                 batch_size=self.hparams.batch_size, sync_dist=True)
        
        self.log('val_epoch/loss', loss, prog_bar=True, 
                 on_epoch=True, add_dataloader_idx=True,
                 batch_size=self.hparams.batch_size, sync_dist=True) 

        self.log('val_epoch/r2', r2, prog_bar=False, 
                 on_epoch=True, add_dataloader_idx=True,
                 batch_size=self.hparams.batch_size, sync_dist=True)
        
        self.log('val_epoch/Spearman_rank', spearman, prog_bar=False, 
                 on_epoch=True, add_dataloader_idx=True,
                 batch_size=self.hparams.batch_size, sync_dist=True)
        self.validation_step_outputs.clear()
        del predictions, targets, rmse, r2, spearman, rmse_avg, loss
        torch.cuda.empty_cache()
        return 
        
    def test_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx, training=False)
        self.test_step_outputs.append(metrics)
        return {
            "predictions": metrics.get("predictions"),
            # "attention_ppi": metrics.get("attention_ppi"),
            "targets": metrics.get("targets")
        }

    def on_test_epoch_end(self):
        # torch.cuda.empty_cache()
        predictions = torch.cat([x.get('predictions') for x in 
                                 self.test_step_outputs], 0)
        targets = torch.cat([x.get('targets') for x in 
                             self.test_step_outputs], 0)
        mse = self.mean_squared_error(predictions, targets)
        rmse = torch.sqrt(mean_squared_error(predictions, targets))

        # calculate R2
        # r2 = 1 - (torch.sum((predictions-target)**2) / torch.sum((target - torch.mean(target))**2))
        r2 = self.r2(predictions, targets)
        # predictions = predictions.type(torch.FloatTensor)
        print(predictions.shape, targets.shape, 
              predictions.get_device(), targets.get_device())
        spearman = self.spearman(predictions, targets)
        self.log('test/r2', r2, prog_bar=False, 
                 on_epoch=True, add_dataloader_idx=True,
                 batch_size=self.hparams.batch_size, sync_dist=True)
        
        self.log('test/rmse', rmse, prog_bar=False, 
                 on_epoch=True, add_dataloader_idx=True,
                 batch_size=self.hparams.batch_size, sync_dist=True)
        
        self.log('test/mse', mse, prog_bar=False, 
                 on_epoch=True, add_dataloader_idx=True,
                 batch_size=self.hparams.batch_size, sync_dist=True)
        
        self.log('test/Spearman_rank', spearman, prog_bar=False, 
                 on_epoch=True, add_dataloader_idx=True,
                 batch_size=self.hparams.batch_size, sync_dist=True)
        del predictions, targets, mse, rmse, r2, spearman
        return 

    def shared_step(self, batch, batch_idx, training=True):
        # TODO send all metrics calculations to CPU
        adj_mat, dist_mat, x = batch[0]
        # print('where are the tensors: \n', x.get_device())
        # print(x.is_pinned(), adj_mat.is_pinned(), dist_mat.is_pinned())
        x_ppi = batch[1].x
        # print(x_ppi.get_device())
        ppi_edge_index = batch[1].edge_index
        ppi_batch = batch[1].batch
        mask = torch.sum(torch.abs(x), dim=-1) != 0
        y_hat, attention_ppi = self.forward(
            x,
            adj_mat,
            dist_mat,
            mask,
            x_ppi,
            ppi_edge_index,
            ppi_batch,
            att_weights=False
        )
        y_hat = y_hat.squeeze(-1)
        loss = self.loss_fn(y_hat, (batch[2]))
        y_targets = batch[2]
        if training:
            rmse = self.RMSE(y_hat, y_targets,squared=False).detach()
            r2 = self.r2(y_hat, y_targets).detach()
            self.log('train_step/RMSE', rmse, prog_bar=False, 
                     on_epoch=False, add_dataloader_idx=True,
                     batch_size=self.hparams.batch_size)
            self.log('train_step/r2', r2, prog_bar=False, 
                     on_epoch=False, add_dataloader_idx=True,
                     batch_size=self.hparams.batch_size)
            self.train_step_outputs.append(
                    {'loss':loss.detach(),
                    'train_rmse_step': rmse, 
                    'train_r2_step': r2})
            del y_hat ,attention_ppi, adj_mat, dist_mat, x, rmse, r2
            del x_ppi, ppi_edge_index, ppi_batch, mask, batch
            return loss 

        else:
            del attention_ppi, adj_mat, dist_mat, x
            del x_ppi, ppi_edge_index, ppi_batch, mask, batch
            return {
            'loss': loss.detach(),
            # 'attention_ppi': attention_ppi,
            'predictions': y_hat,
            'targets': y_targets,
            }

    def configure_optimizers(self):
        opt = Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            amsgrad=True,
            weight_decay = 1e-5,
        )

        sched = {
            'scheduler': ReduceLROnPlateau(
                opt,
                mode='min',
                patience=15,
                factor=0.5,
            ),
            'monitor': 'val_epoch/loss'
        }

        if self.reduce_lr is False:
            return [opt]

        return [opt], [sched]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        version = self.trainer.logger.version[-10:]
        items["v_num"] = version
        return items

    def train_dataloader(self):
        train_dataset = PairDataset(self.data_dir / 
                            'train{}.csv'.format(self.hparams.testing), 
                            self_att=self.hparams.self_att, 
                            test_5000=self.hparams.test_5000)
        print('train dl ', self.hparams.self_att)
        return DataLoader(train_dataset,
                          self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers, 
                          drop_last=True,
                          pin_memory=True,
                          collate_fn=collate_batch)

    def val_dataloader(self):
        val_dataset = PairDataset(self.data_dir / 
                                  'val{}.csv'.format(self.hparams.testing),
                                    self_att=self.hparams.self_att, 
                                    test_5000=self.hparams.test_5000)
        
        return DataLoader(val_dataset,
                          self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=collate_batch)

    def test_dataloader(self):
        test_dataset = PairDataset(self.data_dir / 
                                   'test{}.csv'.format(self.hparams.testing),
                                    self_att=self.hparams.self_att, 
                                    test_5000=self.hparams.test_5000)
        return DataLoader(test_dataset, 
                          self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          collate_fn=collate_batch)


@click.command()
@click.option('--split', default='random', type=click.STRING)
@click.option('--seed', default=42, help='Random seed')
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--epochs', default=300, help='Number of epochs')
@click.option('--gpu', default="0", help='CUDA GPU used for training: 0, 1, 2')
@click.option('--lr', default=1e-3, help='learning rate ')
@click.option('--dataset', default='NCI60DRP', help='dataset name')
@click.option('--new_data', default='', help='New data flag: "Apr", "", ...')
@click.option('--checkpoint', default=None, help='Resume training from checkpoint', type=click.STRING)
@click.option('--testing', default='', help='Random subsample of training set 20 percent flag: "_testing"')
@click.option('--self_att', default='', help='str default "" use self_att "_self_att"')
@click.option('--ppi_depth', default=3, help='depth of gat models 1 2 default:3')
@click.option('--mat_depth', default=8, help='depth of mat transformer default:8')
@click.option('--mat_heads', default=8, help='number of heads of mat transformer default:8')
@click.option('--dim_in_gat', default=5, help='number of heads of mat transformer default:8')
@click.option('--test_5000', default=False, help='Random subsample of molecular graphs 5000 bool: True or False')
@click.option('--profiler', default=None, help='Select a profiler function for debugging: pytorch, or None')
@click.option('--num_workers', default=4, help='Number of workers in dataloader')
@click.option('--gat_model_dim', default=64, help='Dimension of hidden activations in GAT model')

# seed is hardcoded in many places BEWARE
def main(split, batch_size, epochs, seed, gpu, 
         checkpoint, lr, dataset, testing, 
         self_att, ppi_depth, mat_depth, mat_heads, 
         dim_in_gat, test_5000, new_data, profiler, 
         num_workers, gat_model_dim):
    
    conf = Conf(
        lr = lr,
        batch_size = batch_size,
        epochs = epochs,
        reduce_lr = True,
        ckpt_path = checkpoint,
        testing = testing,
        self_att = self_att,
        ppi_depth = ppi_depth,
        mat_depth = mat_depth,
        mat_heads = mat_heads,
        dim_in_gat = dim_in_gat,
        new_data = new_data,
        test_5000 = test_5000,
        profiler = profiler,
        num_workers = num_workers,
        gat_model_dim = gat_model_dim,
        )
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    gc.collect()
    torch.cuda.empty_cache()
    # Not necessary to add self attention to dataset,
    # self_attention flag in GATConv pytorch.geometric module exists
    data_dir = Path(train_test_split(dataset=dataset, 
                    split=split, new_data=new_data)) #, self_att=self_att))  # data seed is 42
    # version needs to come before model or any kind of logging
    
    # TODO below
    # torch.backends.cudnn.benchmark = False
    # CUBLAS_WORKSPACE_CONFIG=':4096:8'
    model = MultimodalAttentionNet(
        conf.to_hparams(),
        data_dir = data_dir,
        # ppi_depth=ppi_depth,
        reduce_lr = conf.reduce_lr,
        dim_in_gat = dim_in_gat,
        ppi_depth = ppi_depth,
        mat_depth = mat_depth,
        mat_heads = mat_heads,
        # add_self_loops=True if self_att == '_self_att' else False
        )

    logger = TensorBoardLogger(
        conf.save_dir,
        name='{}_{}_{}'.format(dataset + new_data, split + self_att, seed),
        version=VERSION)

    # Copy this script and all files used in training
    log_dir = Path(logger.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    print('\n Logging files and tb to: ', str(log_dir))
    shutil.copy(Path(__file__), log_dir)
    shutil.copy(Path(root / "src/train_drp.py"), log_dir)
    shutil.copy(Path(root / "src/create_graphs_mol.py"), log_dir)
    shutil.copy(Path(root / "src/create_graphs_ppi.py"), log_dir)
    shutil.copy(Path(root / "src/multimodal_attention_network.py"), log_dir)
    shutil.copy(Path(root / "src/train_test_split.py"), log_dir)
    early_stop_callback = EarlyStopping(monitor='val_epoch/loss',
                                        min_delta=0.00,
                                        mode='min',
                                        patience=25,
                                        verbose=False)
    
    checkpoint_callback=ModelCheckpoint(
            dirpath=(logger.log_dir + '/checkpoint/'),
            monitor='val_epoch/loss',
            mode='min',
            save_top_k=1,
             save_last=True)

    print("Starting training.")
    trainer = pl.Trainer(
        max_epochs=conf.epochs,
        accelerator='gpu', 
        devices=gpu,
        logger=logger,
        profiler=profiler,
        callbacks=[early_stop_callback, checkpoint_callback],
        deterministic=True,
        # strategy="ddp", 
        accumulate_grad_batches=1,
        precision=16,
        )
    torch.use_deterministic_algorithms(False)
    trainer.fit(model, ckpt_path = conf.ckpt_path)
    results = trainer.test(ckpt_path = 'best')
    results_path = Path(root / "results")

    
    if not results_path.exists():
        results_path.mkdir(exist_ok = True, parents = True)
        with open(results_path / f"drp{new_data}_results.txt", "w") as file:
            file.write("DRP results")
            file.write("\n")

    with open(log_dir / "profiler.txt", "w") as file:
        file.write("Profiler:")
        file.write("\n")
        print(trainer.profiler, file=file)
    
    print("Finished training.\n", results[0] )
    for key, value in results[0].items():
        results[0][key] = round( value,4)
    """
    mse = round(results['test/mse'], 4)
    rmse = round(results['test/rmse'], 4)
    r2 = round(results['test/r2'], 4)
    spearman = round(results['test/Spearman_rank'], 4)
    results = {'Test MSE': mse,
               'Test RMSE': rmse,
               'Test R2': r2,
               'Test Spearman': spearman}
    """
    version = {'version': logger.version}
    results = {logger.name: [version, results, model.hparams]}
    with open(results_path / f"drp{new_data}_results.txt", "a") as file:
        print(results, file=file)
        file.write("\n")
    torch.cuda.empty_cache()
    gc.collect()
    
if __name__ == '__main__':
    main()
