import os
from argparse import Namespace

import numpy as np
import h5py

from datetime import datetime
from collections import defaultdict
from typing import overload, Optional, Union, Tuple, Dict, Callable, List, Any

import torch
import torchvision
import pytorch_lightning as pl
import pylab as plt
from pytorch_lightning.core.memory import ModelSummary
from torch import Tensor
from torch._C import ScriptModule
from torch.optim import Optimizer

from k_space_reconstruction.utils.loss import RAdam
from k_space_reconstruction.utils.metrics import nmse, psnr, ssim, vif, pt_msssim, pt_ssim


class DistributedMetricSum(pl.metrics.Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


class BaseReconstructionModule(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.net = self.get_net(**kwargs)
        self.criterion = kwargs['criterion']
        self.verbose_batch = kwargs['verbose_batch']
        self.optimizer = kwargs['optimizer']
        self.lr = kwargs['lr']
        self.lr_step_size = kwargs['lr_step_size']
        self.lr_gamma = kwargs['lr_gamma']
        self.weight_decay = kwargs['weight_decay']
        args = [s for s in list(kwargs.keys()) if s != 'callbacks']
        self.save_hyperparameters(*args)
        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()

    def forward(self, *args, **kwargs):
        raise NotImplemented

    def get_net(self, **kwargs):
        raise NotImplemented

    def predict(self, batch):
        ks, mask, y, x, mean, std, f_name, slice_id, max_val = batch
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ks, mask, y, x, mean, std, f_name, slice_id, max_val = batch
        yp = self.predict(batch)
        loss = self.criterion(yp, y, mean, std)
        self.log('train_loss_step', loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        ks, mask, y, x, mean, std, f_name, slice_id, max_val = batch
        yp = self.predict(batch)
        loss = self.criterion(yp, y, mean, std)
        return {
            'batch_idx': batch_idx,
            'f_name': f_name,
            'slice_id': slice_id,
            'max_val': max_val,
            'input': x * std + mean,
            'output': yp * std + mean,
            'target': y * std + mean,
            'val_loss': loss
        }

    def validation_step_end(self, val_logs):
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx % self.verbose_batch == 0:
                x = val_logs['input'].cpu().numpy()
                yp = val_logs['output'].cpu().numpy()
                y = val_logs['target'].cpu().numpy()
                grid_yp = torchvision.utils.make_grid(torch.tensor(yp / yp.max((-1, -2), keepdims=True)))
                grid_y = torchvision.utils.make_grid(torch.tensor(y / y.max((-1, -2), keepdims=True)))
                grid_x = torchvision.utils.make_grid(torch.tensor(x / x.max((-1, -2), keepdims=True)))
                self.logger.experiment.add_image('batch_%d/reconstructed' % batch_idx, grid_yp, self.global_step)
                self.logger.experiment.add_image('batch_%d/target' % batch_idx, grid_y, self.global_step)
                self.logger.experiment.add_image('batch_%d/zero_filled' % batch_idx, grid_x, self.global_step)

        nmse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)
        for i, fname in enumerate(val_logs['f_name']):
            slice_id = val_logs['slice_id'][i]
            maxval = val_logs['max_val'][i].cpu().numpy()
            yp = val_logs['output'][i].cpu().numpy()
            y = val_logs['target'][i].cpu().numpy()

            nmse_vals[fname][slice_id] = torch.tensor(nmse(y, yp)).view(1)
            ssim_vals[fname][slice_id] = torch.tensor(ssim(y, yp, maxval=maxval)).view(1)
            psnr_vals[fname][slice_id] = torch.tensor(psnr(y, yp)).view(1)
        return {
            'val_loss': val_logs['val_loss'],
            'nmse_vals': nmse_vals,
            'ssim_vals': ssim_vals,
            'psnr_vals': psnr_vals,
        }

    def validation_epoch_end(self, val_logs):
        losses = []
        nmse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)

        for val_log in val_logs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["nmse_vals"].keys():
                nmse_vals[k].update(val_log["nmse_vals"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["psnr_vals"].keys():
                psnr_vals[k].update(val_log["psnr_vals"][k])
        # check to make sure we have all files in all metrics
        assert nmse_vals.keys() == ssim_vals.keys() == psnr_vals.keys()

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in nmse_vals.keys():
            local_examples = local_examples + 1
            metrics["nmse"] = metrics["nmse"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals[fname].items()])
            )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )
            metrics["psnr"] = metrics["psnr"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log('val_loss', val_loss / tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(metric, value / tot_examples)

    def test_step(self, batch, batch_idx):
        ks, mask, y, x, mean, std, f_name, slice_id, max_val = batch
        yp = self.predict(batch)
        return {
            'f_name': f_name,
            'slice_id': slice_id,
            'output': (yp * std + mean).cpu().numpy()
        }

    def test_epoch_end(self, test_logs) -> None:
        outputs = defaultdict(dict)

        for log in test_logs:
            for i, (fname, slice_id) in enumerate(zip(log['f_name'], log['slice_id'])):
                outputs[fname][slice_id] = log['output'][i]

        for fname in outputs:
            outputs[fname] = np.stack([out for _, out in sorted(outputs[fname].items())])
        # TODO: make args for it
        save_file = os.path.join(self.trainer.default_root_dir, str(datetime.now()) + '.h5')
        with h5py.File(save_file, 'w') as hf:
            for fname in outputs:
                hf.create_dataset(os.path.basename(fname), data=outputs[fname])

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'AdamW':
            optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()))
        elif self.optimizer == 'SGD':
            optim = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'RMSprop':
            optim = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'RAdam':
            optim = RAdam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError
        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.lr_step_size, self.lr_gamma)
        return [optim], [scheduler]


class DummyModule(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(DummyModule, self).__init__(**kwargs)

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(1)

    def get_net(self, **kwargs):
        return torch.nn.Sequential()
