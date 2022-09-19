import os
import random
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.utils.data
import pytorch_lightning as pl
import torch.nn.functional as F

from networks import Unet, UNet3d, AttU_Net, UnetCS, CascadeCS

from optimize_pattern import PatternSampler, GridSampler
from optimize_pattern import ce_loss, dice_loss
from optimize_pattern import get_dataset, COIL_MAP
from optimize_pattern import MetricFuncGen, _FfuncGen
from optimize_pattern import pt_ssim, refine_data_batch

import sys
from pytorch_lightning.callbacks.progress import ProgressBarBase
from tqdm import tqdm


torch.manual_seed(228)
random.seed(228)
np.random.seed(228)


class GlobalProgressBar(ProgressBarBase):

    def __init__(self, process_position: int = 0):
        super().__init__()
        self._process_position = process_position
        self._enabled = True
        self.main_progress_bar = None

    def __getstate__(self):
        # can't pickle the tqdm objects
        state = self.__dict__.copy()
        state['main_progress_bar'] = None
        return state

    @property
    def process_position(self) -> int:
        return self._process_position

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.main_progress_bar = tqdm(
            desc='Total Epochs',
            initial=trainer.current_epoch,
            total=trainer.max_epochs,
            position=self.process_position,
            disable=False,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )

    def on_train_end(self, trainer, pl_module):
        self.main_progress_bar.close()

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        super(GlobalProgressBar, self).on_validation_epoch_end(trainer, pl_module)
        self.main_progress_bar.update(1)


class plReconstructionUnetModule(pl.LightningModule):

    def __init__(self, unet_params, ksampler_params, args_params, loss_fn, lr=3e-4, optim='adam',
                 verbose_batch=100, f_pattern=None, grid=False, model=False):
        super().__init__()
        self.net = UnetCS(**unet_params)
        if model:
            self.net.load_state_dict(torch.load(model, map_location='cpu'))
        if grid:
            self.k_sampler = GridSampler(**ksampler_params)
        else:
            self.k_sampler = PatternSampler(**ksampler_params)
        self.lr = lr
        self.optim = optim
        self.args_params = args_params
        self.loss: MetricFuncGen = loss_fn
        # Load pattern
        if f_pattern:
            self.pattern = torch.load(f_pattern, map_location=torch.device('cpu'))
        else:
            self.pattern = None
        self.verbose_batch = verbose_batch
        self.save_hyperparameters()

    def forward(self, batch):
        return self.net(batch)

    def configure_optimizers(self):
        if self.optim == 'adam':
            return torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-6)
        elif self.optim == 'adamw':
            return torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        else:
            raise ValueError

    def training_step(self, batch, batch_idx):
        _batch = refine_data_batch(batch)
        if self.k_sampler:
            gt_batch = self.k_sampler(_batch)
            batch = self.k_sampler(_batch, self.pattern)
        else:
            gt_batch = {k:v.clone() for k,v in _batch.items()}
        loss = self.loss(y=batch, x=gt_batch, s_func=self)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _batch = refine_data_batch(batch)
        if self.k_sampler:
            gt_batch = self.k_sampler(_batch)
            batch = self.k_sampler(_batch, self.pattern)
        else:
            gt_batch = {k:v.clone() for k,v in _batch.items()}
        loss = self.loss(y=batch, x=gt_batch, s_func=self)
        self.log('val_loss', loss)
        if batch_idx % self.verbose_batch == 0:
            p = self.forward(batch)
            img = (batch['img'] * batch['std'] + batch['mean'])
            p = (p * batch['std'] + batch['mean']).detach().cpu()
            for c in range(img.shape[1]):
                self.logger.experiment.add_images(
                    f'image{c}/{batch_idx}',
                    (img[:, c] / img[:, c].max())[:, None].cpu(),
                    self.current_epoch
                )
                self.logger.experiment.add_images(
                    f'pred{c}/{batch_idx}',
                    (p[:, c] / p[:, c].max())[:, None].cpu(),
                    self.current_epoch
                )
        return loss


class plReconstructionCascadeModule(CascadeCS, pl.LightningModule):

    def __init__(self, chans, num_pool_layers, ksampler_params, args_params, loss_fn, lr=3e-4, optim='adam',
                 verbose_batch=100, f_pattern=None, grid=False, model=False):
        super(plReconstructionCascadeModule, self).__init__(chans=chans, num_pool_layers=num_pool_layers)
        if model:
            self.net.load_state_dict(torch.load(model, map_location='cpu'))
        if grid:
            self.k_sampler = GridSampler(**ksampler_params)
        else:
            self.k_sampler = PatternSampler(**ksampler_params)
        self.lr = lr
        self.optim = optim
        self.args_params = args_params
        self.loss: MetricFuncGen = loss_fn
        # Load pattern
        if f_pattern:
            self.pattern = torch.load(f_pattern, map_location=torch.device('cpu'))
        else:
            self.pattern = None
        self.verbose_batch = verbose_batch
        self.save_hyperparameters()

    def configure_optimizers(self):
        if self.optim == 'adam':
            return torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-6)
        elif self.optim == 'adamw':
            return torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        else:
            raise ValueError

    def training_step(self, batch, batch_idx):
        _batch = refine_data_batch(batch)
        if self.k_sampler:
            gt_batch = self.k_sampler(_batch)
            batch = self.k_sampler(_batch, self.pattern)
        else:
            gt_batch = {k:v.clone() for k,v in _batch.items()}
        loss = self.loss(y=batch, x=gt_batch, s_func=self)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _batch = refine_data_batch(batch)
        if self.k_sampler:
            gt_batch = self.k_sampler(_batch)
            batch = self.k_sampler(_batch, self.pattern)
        else:
            gt_batch = {k:v.clone() for k,v in _batch.items()}
        loss = self.loss(y=batch, x=gt_batch, s_func=self)
        self.log('val_loss', loss)
        if batch_idx % self.verbose_batch == 0:
            p = self.forward(batch)
            img = (batch['img'] * batch['std'] + batch['mean'])
            p = (p * batch['std'] + batch['mean']).detach().cpu()
            for c in range(img.shape[1]):
                self.logger.experiment.add_images(
                    f'image{c}/{batch_idx}',
                    (img[:, c] / img[:, c].max())[:, None].cpu(),
                    self.current_epoch
                )
                self.logger.experiment.add_images(
                    f'pred{c}/{batch_idx}',
                    (p[:, c] / p[:, c].max())[:, None].cpu(),
                    self.current_epoch
                )
        return loss



class plUnetModule(pl.LightningModule):
    
    def __init__(self, unet_params, ksampler_params, args_params, loss_fn, lr=3e-4, optim='adam',
                 verbose_batch=100, f_pattern=None, grid=False, model=False):
        super().__init__()
        self.net = Unet(**unet_params)
        if model:
            self.net.load_state_dict(torch.load(model, map_location='cpu'))
        if grid:
            self.k_sampler = GridSampler(**ksampler_params)
        else:
            self.k_sampler = PatternSampler(**ksampler_params)
        self.lr = lr
        self.optim = optim
        self.args_params = args_params
        self.loss: MetricFuncGen = loss_fn
        # Load pattern
        if f_pattern:
            self.pattern = torch.load(f_pattern, map_location=torch.device('cpu'))
        else:
            self.pattern = None
        self.verbose_batch = verbose_batch
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.net(x)
    
    def configure_optimizers(self):
        if self.optim == 'adam':
            return torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-6)
        elif self.optim == 'adamw':
            return torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        else:
            raise ValueError

    def training_step(self, batch, batch_idx):
        if self.k_sampler:
            batch = self.k_sampler(batch, self.pattern)
        loss = self.loss(y=batch, x=batch, s_func=self)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.k_sampler:
            batch = self.k_sampler(batch, self.pattern)
        loss = self.loss(y=batch, x=batch, s_func=self)
        self.log('val_loss', loss)
        if batch_idx % self.verbose_batch == 0:
            p = self(batch['img'])
            img = (batch['img'] * batch['std'] + batch['mean'])[0]
            p = p[0].detach().cpu().softmax(0)
            pmask = torch.zeros(batch['mask'].shape[-2:])
            for i in range(p.shape[0]):
                pmask += p[i] * (i + 1)
            for i in range(img.shape[0]):
                self.logger.experiment.add_image(
                    f'image{i}/{batch_idx}', 
                    (img[i] / img[i].max()).cpu()[None],
                    self.current_epoch
                )
            self.logger.experiment.add_image(
                f'pred/{batch_idx}', 
                (pmask / pmask.max()).cpu()[None],
                self.current_epoch
            )
            self.logger.experiment.add_image(
                f'gt/{batch_idx}', 
                (batch['mask'][0][0] / batch['mask'][0][0].max()).cpu()[None],
                self.current_epoch
            )
        return loss


class plUnetAttModule(pl.LightningModule):

    def __init__(self, unet_params, ksampler_params, args_params, loss_fn, lr=3e-4, optim='adam',
                 verbose_batch=100, f_pattern=None, grid=False, model=False):
        super().__init__()
        self.net = AttU_Net(**unet_params)
        if model:
            self.net.load_state_dict(torch.load(model, map_location='cpu'))
        if grid:
            self.k_sampler = GridSampler(**ksampler_params)
        else:
            self.k_sampler = PatternSampler(**ksampler_params)
        self.lr = lr
        self.optim = optim
        self.args_params = args_params
        self.loss: MetricFuncGen = loss_fn
        # Load pattern
        if f_pattern:
            self.pattern = torch.load(f_pattern, map_location=torch.device('cpu'))
        else:
            self.pattern = None
        self.verbose_batch = verbose_batch
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        if self.optim == 'adam':
            return torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-6)
        elif self.optim == 'adamw':
            return torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        else:
            raise ValueError

    def training_step(self, batch, batch_idx):
        if self.k_sampler:
            batch = self.k_sampler(batch, self.pattern)
        loss = self.loss(y=batch, x=batch, s_func=self)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.k_sampler:
            batch = self.k_sampler(batch, self.pattern)
        loss = self.loss(y=batch, x=batch, s_func=self)
        self.log('val_loss', loss)
        if batch_idx % self.verbose_batch == 0:
            p = self(batch['img'])
            img = (batch['img'] * batch['std'] + batch['mean'])[0]
            p = p[0].detach().cpu().softmax(0)
            pmask = torch.zeros(batch['mask'].shape[-2:])
            for i in range(p.shape[0]):
                pmask += p[i] * (i + 1)
            for i in range(img.shape[0]):
                self.logger.experiment.add_image(
                    f'image{i}/{batch_idx}',
                    (img[i] / img[i].max()).cpu()[None],
                    self.current_epoch
                )
            self.logger.experiment.add_image(
                f'pred/{batch_idx}',
                (pmask / pmask.max()).cpu()[None],
                self.current_epoch
            )
            self.logger.experiment.add_image(
                f'gt/{batch_idx}',
                (batch['mask'][0][0] / batch['mask'][0][0].max()).cpu()[None],
                self.current_epoch
            )
        return loss


class plUnet3DModule(pl.LightningModule):

    def __init__(self, unet_params, ksampler_params, args_params, loss_fn, lr=3e-4, optim='adam',
                 verbose_batch=100, f_pattern=None, grid=False, model=False):
        super(plUnet3DModule, self).__init__()
        self.net = UNet3d(**unet_params)
        if model:
            self.net.load_state_dict(torch.load(model, map_location='cpu'))
        if grid:
            self.k_sampler = GridSampler(**ksampler_params)
        else:
            self.k_sampler = PatternSampler(**ksampler_params)
        self.lr = lr
        self.optim = optim
        self.args_params = args_params
        self.save_hyperparameters()
        self.logged_images = 0
        # Load pattern
        if f_pattern:
            self.pattern = torch.load(f_pattern, map_location=torch.device('cpu'))
        else:
            self.pattern = None
        self.verbose_batch = verbose_batch
        self.loss: MetricFuncGen = loss_fn

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        if self.optim == 'adam':
            return torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-6)
        elif self.optim == 'adamw':
            return torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        else:
            raise ValueError

    def training_step(self, batch, batch_idx):
        if self.k_sampler:
            batch = self.k_sampler(batch, self.pattern)
        loss = self.loss(y=batch, x=batch, s_func=self)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.k_sampler:
            batch = self.k_sampler(batch, self.pattern)
        loss = self.loss(y=batch, x=batch, s_func=self)
        self.log('val_loss', loss)
        if self.logged_images < 1:
            p = self(batch['img'])
            p = p.movedim(1, 2).flatten(0, 1).detach().cpu().softmax(1)
            mask = batch['mask'].movedim(1, 2).flatten(0, 1).cpu()[:, 0]
            img = (batch['img'] * batch['std'] + batch['mean']).movedim(1, 2).flatten(0,1).cpu()
            pmask = p.softmax(1).argmax(1)
            for i in range(img.shape[1]):
                self.logger.experiment.add_images(
                    f'image{i}/{batch_idx}',
                    (img[:,i] / img[:,i].view(img.shape[0], -1).max(1).values[:, None, None]).cpu()[:, None],
                    self.current_epoch
                )
            self.logger.experiment.add_images(
                f'pred/{batch_idx}',
                (pmask / pmask.view(pmask.shape[0], -1).max(1).values[:, None, None]).cpu()[:, None],
                self.current_epoch
            )
            self.logger.experiment.add_images(
                f'gt/{batch_idx}',
                (mask / mask.view(mask.shape[0], -1).max(1).values[:, None, None]).cpu()[:, None],
                self.current_epoch
            )
            self.logged_images += 1
        return loss

    def on_epoch_start(self) -> None:
        self.logged_images = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='acdc',
        choices=['acdc', 'brats', 'fastmri', 'fastmri_plus'],
        type=str,
        required=True
    )
    parser.add_argument(
        '--batch',
        default=32,
        type=int,
    )
    parser.add_argument(
        '--val_batch',
        default=32,
        type=int,
    )
    parser.add_argument(
        '--gpus',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--multicoil',
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
    )
    parser.add_argument(
        '--preload',
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
    )
    parser.add_argument(
        '--grid',
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
    )
    parser.add_argument(
        '--pattern',
        default=None,
        type=str
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['unet', 'unet-att', 'unet3d', 'none', 'cs-unet', 'cs-cascade']
    )
    parser.add_argument(
        '--model_path',
        default=None,
        type=str
    )
    parser.add_argument(
        '--epoch',
        default=100,
        type=int,
    )
    parser.add_argument(
        '--lr',
        default=3e-4,
        type=float,
    )
    parser.add_argument(
        '--loss',
        type=str,
        choices=[
            'dice', 'dice_ce', 'ce',
            'ssim', 'l1', 'ssim-l1',
            'ssim2', 'ssim2-l1',
            'plus-l1', 'plus-ssim', 'plus-ssim-l1',
            'plus-ssim2', 'plus-ssim2-l1',
        ],
        default='dice',
    )
    parser.add_argument(
        '--optim',
        default='adam',
        type=str,
        choices=['adam', 'adamw']
    )
    parser.add_argument(
        '--nchans',
        default=16,
        type=int
    )
    parser.add_argument(
        '--nlayers',
        default=4,
        type=int
    )
    parser.add_argument(
        '--test',
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
    )
    parser.add_argument(
        '--test_results',
        default=None,
        type=str,
    )
    parser.add_argument(
        '--verbose',
        default=25,
        type=int
    )
    args = parser.parse_args()
    # Load datasets
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset, val_dataset = get_dataset(
        args.dataset, device,
        preload=args.preload,
        volumetric=(args.test or (args.model == 'unet3d')) and args.model not in ['cs-unet', 'cs-cascade']
    )
    train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1 if args.test else args.val_batch, shuffle=False)

    if args.dataset == 'fastmri_plus':
        train_generator = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=lambda batch: [{k:v for k,v in b.items()} for b in batch],
            batch_size=args.batch, shuffle=True)
        val_generator = torch.utils.data.DataLoader(
            val_dataset,
            collate_fn=lambda batch: [{k:v for k,v in b.items()} for b in batch],
            batch_size=1 if args.test else args.val_batch, shuffle=False)
    else:
        train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1 if args.test else args.val_batch, shuffle=False)

    if args.dataset == 'acdc':
        ic = 1
        oc = 4
    elif args.dataset == 'brats':
        ic = 4
        oc = 4
    elif args.dataset.startswith('fastmri'):
        ic = 1
        oc = 1
    else:
        raise ValueError
    # Get image sizes
    tmp_data = val_dataset[0]
    size_x, size_y = tmp_data['img'].shape[-2:]
    # Set k-space sampler params
    f_map = COIL_MAP[args.dataset] if args.multicoil else None
    # Set metrics and loss function generators
    if args.model == 'unet3d':
        metric_funcs = dict(dice=MetricFuncGen('dice_3d'), ssim=MetricFuncGen('ssim_3d'))
        loss_func = MetricFuncGen(args.loss + '_3d')
    elif args.dataset == 'fastmri' and args.model == 'none':
        metric_funcs = dict(ssim=MetricFuncGen('ssim'))
        loss_func = MetricFuncGen(args.loss)
    elif args.model in ['cs-unet', 'cs-cascade']:
        if args.dataset == 'fastmri_plus':
            metric_funcs = dict(ssim_attr=MetricFuncGen('cs-plus-ssim'), ssim=MetricFuncGen('cs-ssim'))
            loss_func = MetricFuncGen('cs-' + str(args.loss))
        else:
            metric_funcs = dict(ssim=MetricFuncGen('cs-ssim'))
            loss_func = MetricFuncGen('cs-' + str(args.loss))
    else:
        metric_funcs = dict(dice=MetricFuncGen('dice'), ssim=MetricFuncGen('ssim'))
        loss_func = MetricFuncGen(args.loss)
    # Prepare PL module
    if args.model == 'unet':
        model = plUnetModule(
            unet_params=dict(in_chans=ic, out_chans=oc, chans=args.nchans, num_pool_layers=args.nlayers),
            ksampler_params=dict(f_maps=f_map, shape=[size_x, size_y], device=device),
            args_params=args.__dict__,
            loss_fn=loss_func,
            lr=args.lr,
            optim=args.optim,
            f_pattern=args.pattern,
            grid=args.grid,
            model=args.model_path,
            verbose_batch=args.verbose
        )
    elif args.model == 'unet-att':
        model = plUnetAttModule(
            unet_params=dict(img_ch=ic, output_ch=oc, channels=args.nchans),
            ksampler_params=dict(f_maps=f_map, shape=[size_x, size_y], device=device),
            args_params=args.__dict__,
            loss_fn=loss_func,
            lr=args.lr,
            optim=args.optim,
            f_pattern=args.pattern,
            grid=args.grid,
            model=args.model_path,
            verbose_batch=args.verbose
        )
    elif args.model == 'cs-unet':
        model = plReconstructionUnetModule(
            unet_params=dict(in_chans=ic, out_chans=oc, chans=args.nchans, num_pool_layers=args.nlayers),
            ksampler_params=dict(f_maps=f_map, shape=[size_x, size_y], device=device),
            args_params=args.__dict__,
            loss_fn=loss_func,
            lr=args.lr,
            optim=args.optim,
            f_pattern=args.pattern,
            grid=args.grid,
            model=args.model_path,
            verbose_batch=args.verbose
        )
    elif args.model == 'cs-cascade':
        model = plReconstructionCascadeModule(
            chans=args.nchans, num_pool_layers=args.nlayers,
            ksampler_params=dict(f_maps=f_map, shape=[size_x, size_y], device=device),
            args_params=args.__dict__,
            loss_fn=loss_func,
            lr=args.lr,
            optim=args.optim,
            f_pattern=args.pattern,
            grid=args.grid,
            model=args.model_path,
            verbose_batch=args.verbose
        )
    elif args.model == 'unet3d':
        model = plUnet3DModule(
            unet_params=dict(in_channels=ic, n_classes=oc, n_channels=args.nchans),
            ksampler_params=dict(f_maps=f_map, shape=[size_x, size_y], device=device),
            args_params=args.__dict__,
            loss_fn=loss_func,
            lr=args.lr,
            optim=args.optim,
            f_pattern=args.pattern,
            grid=args.grid,
            model=args.model_path,
            verbose_batch=args.verbose
        )
    elif args.model == 'none':
        model = plUnetModule(
            unet_params=dict(in_chans=ic, out_chans=oc, chans=args.nchans, num_pool_layers=args.nlayers),
            ksampler_params=dict(f_maps=f_map, shape=[size_x, size_y], device=device),
            args_params=args.__dict__,
            loss_fn=loss_func,
            lr=args.lr,
            optim=args.optim,
            f_pattern=args.pattern,
            grid=args.grid,
            model=args.model_path,
            verbose_batch=args.verbose
        )
        model.net = torch.nn.Identity()
    else:
        raise ValueError
    if not args.test:
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                save_last=True, save_top_k=7,
                save_weights_only=False,monitor='val_loss', filename='{epoch}-{val_loss:.5f}'
            ),
            # GlobalProgressBar()
        ]
        # Init trainer
        trainer = pl.Trainer(
            gpus=args.gpus,
            accelerator='ddp' if args.gpus > 1 else None,
            terminate_on_nan=True,
            max_epochs=args.epoch,
            callbacks=callbacks,
            default_root_dir=f'logs/models/{args.dataset}_{"multi" if args.multicoil else "single"}coil/{args.model}'
        )
        trainer.fit(model, train_generator, val_generator)
    else:
        model.net = model.net.to(device)
        model.eval()
        results = defaultdict(list)
        if args.model == 'unet3d':
            metric_funcs = dict(dice=MetricFuncGen('dice_3d'), dice_vec=MetricFuncGen(f'dice_3d_{args.dataset}_3d_vec'), ssim=MetricFuncGen('ssim_3d'))
        elif args.model == 'none':
            metric_funcs = dict(ssim=MetricFuncGen('ssim_3d'))
        elif args.model in ['cs-unet', 'cs-cascade']:
            if args.dataset == 'fastmri_plus':
                metric_funcs = dict(ssim_attr=MetricFuncGen('cs-plus-ssim'), ssim=MetricFuncGen('cs-ssim'))
            else:
                metric_funcs = dict(ssim=MetricFuncGen('cs-ssim'))
        else:
            metric_funcs = dict(dice=MetricFuncGen('dice_322d'), dice_vec=MetricFuncGen(f'dice_3d_{args.dataset}_vec'), ssim=MetricFuncGen('ssim_322d'))
        for data in tqdm(val_generator, total=len(val_generator)):
            data = refine_data_batch(data)
            if not args.dataset.startswith('fastmri') and torch.any(data['mask']):
                scalar_funcs = {
                    metric: _FfuncGen(data, model.net, model.k_sampler, f)
                    for metric, f in metric_funcs.items()
                }
                for metric_name, metric_func in scalar_funcs.items():
                    if metric_name.endswith('_vec'):
                        dice_scores = metric_func(model.pattern)
                        for i, score in enumerate(dice_scores):
                            results[f'{metric_name}_{i}'].append(score.item())
                    else:
                        results[metric_name].append(metric_func(model.pattern).item())
            elif args.dataset.startswith('fastmri'):
                scalar_funcs = {
                    metric: _FfuncGen(data, model, model.k_sampler, f)
                    for metric, f in metric_funcs.items()
                }
                for metric_name, metric_func in scalar_funcs.items():
                    if metric_name.endswith('_vec'):
                        dice_scores = metric_func(model.pattern)
                        for i, score in enumerate(dice_scores):
                            results[f'{metric_name}_{i}'].append(score.item())
                    else:
                        results[metric_name].append(metric_func(model.pattern).item())
        df = pd.DataFrame.from_dict({c:v for c,v in results.items()})
        if args.test_results:
            df.to_csv(args.test_results, sep='\t')
        else:
            print(df.describe())

