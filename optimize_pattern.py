import argparse
import logging
import math
import os
import io
import cv2
import pickle
import warnings
import random
from collections import defaultdict
from functools import reduce
from typing import Callable, Dict, List, Tuple

import h5py
import numpy as np
import pylab as plt
import pandas as pd
from sklearn.model_selection import KFold

import torch
import torch.nn.functional as F
import torch.utils.data

from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from networks import Unet, UNet3d, AttU_Net, UnetCS, CascadeCS

from kspace import pt_kspace2spatial as IFt
from kspace import pt_spatial2kspace as Ft
from pytorch_msssim import ssim
from typing import Set, List, Tuple

import sys
sys.path.append('./pytorch_nufft')
import nufft
from configs import *
# plt.style.use('dark_background')

torch.manual_seed(228)
random.seed(228)
np.random.seed(228)


def root_sum_of_squares(images, coil_dim=1):
    return images.square().sum(coil_dim).sqrt()


def t2i(x):
    x = x - x.min()
    x = x / x.max()
    return x


def ce_loss(true, logits, weights, ignore=255):
    ce_loss = F.cross_entropy(
        logits.float(),
        true.long(),
        ignore_index=ignore,
        weight=weights,
    )
    return ce_loss


from math import exp


def pt_ssim2(img1, img2, window_size=7, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def pt_ssim(pred, gt):
    from pytorch_msssim import ssim
    return ssim(t2i(pred), t2i(gt), win_size=11, data_range=1.0)

def dice_loss(gt: torch.tensor, logits: torch.tensor, eps=1e-7):
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[gt.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[gt.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, gt.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return 1 - dice_loss


def dice_coeffs(gt: torch.tensor, logits: torch.tensor):
    num_classes = logits.shape[1]
    probas = F.softmax(logits, dim=1)
    probas[probas > 0.5] = 1; probas[probas <= 0.5] = 0
    pmask = torch.zeros_like(gt).float()
    for i in range(1, num_classes):
        pmask[:,0] += i * probas[:,i]
    dice_ls = []
    for i in range(1, num_classes):
        yt = (gt == i).float().flatten()
        yp = (pmask==i).float().flatten()
        intersection = torch.sum(yt * yp)
        cardinality = torch.sum(yt + yp)
        dice_ls.append((2. * intersection / (cardinality + 1e-7)))
    return dice_ls


def dice_3d_acdc_vec(gt: torch.tensor, logits: torch.tensor, eps=1e-7):
    pmask = torch.nn.functional.softmax(logits, dim=1).argmax(1)[:, None]
    dice_ls = []
    # 1
    true_1_hot = (gt == 1).float().flatten()
    pred_1_hot = (pmask==1).float().flatten()
    intersection = torch.sum(pred_1_hot * true_1_hot)
    cardinality = torch.sum(pred_1_hot + true_1_hot)
    dice_ls.append((2. * intersection / (cardinality + eps)))
    # 2
    true_1_hot = (gt == 2).float().flatten()
    pred_1_hot = (pmask==2).float().flatten()
    intersection = torch.sum(pred_1_hot * true_1_hot)
    cardinality = torch.sum(pred_1_hot + true_1_hot)
    dice_ls.append((2. * intersection / (cardinality + eps)))
    # 3
    true_1_hot = (gt == 3).float().flatten()
    pred_1_hot = (pmask==3).float().flatten()
    intersection = torch.sum(pred_1_hot * true_1_hot)
    cardinality = torch.sum(pred_1_hot + true_1_hot)
    dice_ls.append((2. * intersection / (cardinality + eps)))
    return dice_ls

def dice_3d_brats_vec(gt: torch.tensor, logits: torch.tensor, eps=1e-7):
    pmask = torch.nn.functional.softmax(logits, dim=1).argmax(1)[:, None]
    dice_ls = []
    # WT
    true_1_hot = ((gt == 1) | (gt == 2) | (gt == 3)).float().flatten()
    pred_1_hot = ((pmask==1) | (pmask==2) | (pmask==3)).float().flatten()
    intersection = torch.sum(pred_1_hot * true_1_hot)
    cardinality = torch.sum(pred_1_hot + true_1_hot)
    dice_ls.append((2. * intersection / (cardinality + eps)))
    # TC
    true_1_hot = ((gt == 1) | (gt == 3)).float().flatten()
    pred_1_hot = ((pmask==1) | (pmask==3)).float().flatten()
    intersection = torch.sum(pred_1_hot * true_1_hot)
    cardinality = torch.sum(pred_1_hot + true_1_hot)
    dice_ls.append((2. * intersection / (cardinality + eps)))
    # ET
    true_1_hot = ((gt == 1)).float().flatten()
    pred_1_hot = ((pmask==1)).float().flatten()
    intersection = torch.sum(pred_1_hot * true_1_hot)
    cardinality = torch.sum(pred_1_hot + true_1_hot)
    dice_ls.append((2. * intersection / (cardinality + eps)))
    return dice_ls


class _SummaryWriter(SummaryWriter):

    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix=''):
        super(_SummaryWriter, self).__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)

    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):

        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)
        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            if v is not None:
                self.add_scalar(k, v)


class PatternSampler:

    def __init__(self, f_maps, shape, device=torch.device('cpu')) -> None:
        self.multi_coil = bool(f_maps)
        self.device = device
        if self.multi_coil:
            self.smap = torch.load(f_maps).to(device)
            self.grid = torch.stack(
                torch.meshgrid(
                    torch.arange(-shape[0]/2, shape[0]/2),
                    torch.arange(-shape[1]/2, shape[1]/2))
            ).flatten(1).to(device)

    def _get_k_space(self, batch, sampling):
        img = batch['img'] * batch['std'] + batch['mean']
        if len(img.shape) == 5:
            img = img.movedim(1, 2).flatten(0, 1)
        _device = img.device
        if self.multi_coil:
            img = img[:, None] * self.smap[None, :, None].to(img.device)
            ks = nufft.nufft(img, self.grid.T.to(img.device), device=img.device).reshape(img.shape)
        else:
            ks = Ft(img)
        return ks

    def _get_sampled_image(self, batch, sampling):
        ks = self._get_k_space(batch, sampling)
        if sampling is not None:
            if self.multi_coil:
                ks = ks * sampling[None].to(ks.device)
            else:
                ks = ks * sampling.to(ks.device)
        simg = IFt(ks).abs()
        batch['ks'] = ks
        if self.multi_coil:
            simg = root_sum_of_squares(simg, coil_dim=1)
        return simg

    def __call__(self, batch, sampling=None):
        sampled_img = self._get_sampled_image(batch, sampling)
        if len(batch['img'].shape) > 4:
            sampled_img = sampled_img.reshape_as(batch['img'].movedim(1, 2)).movedim(1, 2)
        if 'IGS_BRATS_RT' in os.environ and os.environ['IGS_BRATS_RT'] == '1':
            sampled_img = sampled_img.movedim(-1, -2)
            mask = batch['mask'].movedim(-1, -2) if 'mask' in batch else None
        else:
            mask = batch['mask'] if 'mask' in batch else None
        s, m = torch.std_mean(sampled_img, [-1, -2])
        s = s.unsqueeze(-1).unsqueeze(-1)
        m = m.unsqueeze(-1).unsqueeze(-1)
        sampled_img = (sampled_img - m) / (s + 1e-11)
        return dict(
            mask=mask.to(sampled_img.device) if 'mask' in batch and batch['mask'] is not None else None,
            img=sampled_img.to(sampled_img.device),
            mean=m.to(sampled_img.device),
            std=s.to(sampled_img.device),
            sampling=sampling.to(sampled_img.device) if sampling is not None else None,
            ks = batch['ks'].to(sampled_img.device),
            boxes=batch['boxes'] if 'boxes' in batch and batch['boxes'] is not None else None
        )


class GridSampler(PatternSampler):

    def __init__(self, f_maps, shape, device=torch.device('cpu')) -> None:
        super(GridSampler, self).__init__(f_maps, shape, device)

    def _get_k_space(self, batch, sampling):
        img = batch['img'] * batch['std'] + batch['mean']
        if len(img.shape) == 5:
            img = img.movedim(1,2).flatten(0, 1)
        _device = img.device
        if self.multi_coil:
            img = img[:, None] * self.smap[None, :, None]
        ks = nufft.nufft(img, sampling.T, device=device).reshape(img.shape[0], img.shape[1], -1)
        return ks

    def _get_sampled_image(self, batch, sampling):
        img_shape = batch['img'].shape
        ks = self._get_k_space(batch, sampling)
        if sampling is not None:
            if self.multi_coil:
                out_shape = [*ks.shape[:3], *img_shape[-2:]]
            else:
                out_shape = [*ks.shape[:2], *img_shape[-2:]]
            sampled_img = nufft.nufft_adjoint(ks, sampling.T, out_shape=out_shape, device=ks.device).abs()
        else:
            sampled_img = IFt(ks).abs()
        if self.multi_coil:
            sampled_img = root_sum_of_squares(sampled_img, coil_dim=1)
        return sampled_img


class PatternGridSampler(GridSampler):

    def __init__(self, f_maps, shape, pattern: torch.tensor, device=torch.device('cpu')) -> None:
        super(PatternGridSampler, self).__init__(f_maps, shape, device)
        self.pattern = pattern
        self.grid = torch.stack(
            torch.meshgrid(
                torch.arange(-shape[0] / 2, shape[0] / 2),
                torch.arange(-shape[1] / 2, shape[1] / 2))
        ).flatten(1).to(device)

    def _get_k_space(self, batch, sampling):
        img = batch['img'] * batch['std'] + batch['mean']
        if len(img.shape) == 5:
            img = img.movedim(1,2).flatten(0, 1)
        _device = img.device
        if self.multi_coil:
            img = img[:, None] * self.smap[None, :, None]
        ks = nufft.nufft(img, self.grid.T, device=device).reshape(*img.shape)
        mask = torch.ones_like(img)
        if self.multi_coil:
            mask = mask * self.pattern[None].to(ks.device)
            mask = mask[0, 0, 0]
            ks = ks.view(*ks.shape[:3], -1)[:, :, :, mask.flatten() == 1]
            return ks
        else:
            mask = mask * self.pattern.to(device)
            mask = mask[0, 0]
            ks = ks.view(*ks.shape[:2], -1)[:, :, mask.flatten() == 1]
            return ks

    def _get_sampled_image(self, batch, sampling):
        img_shape = batch['img'].shape
        ks = self._get_k_space(batch, sampling)
        if self.multi_coil:
            out_shape = [*ks.shape[:3], *img_shape[-2:]]
        else:
            out_shape = [*ks.shape[:2], *img_shape[-2:]]
        sampled_img = nufft.nufft_adjoint(ks, sampling.T, out_shape=out_shape, device=ks.device).abs()
        if self.multi_coil:
            sampled_img = root_sum_of_squares(sampled_img, coil_dim=1)
        return sampled_img


class PatternOptimizer:

    def __init__(
            self,
            model: torch.nn.Module,
            acceleration: float,
            img_shape: List[int],
            dimensions: int,
            device: torch.DeviceObjType,
            log_dir: str,
            comment: str,
    ) -> None:
        assert dimensions in [1, 2]
        assert len(img_shape) == 2
        self.model = model.eval()
        self.method = self.__class__.__name__
        self.acceleration = acceleration
        self.image_shape = img_shape
        self.dimensions = dimensions
        self.device = device
        self.log_dir = log_dir
        self.comment = comment
        self.writer = _SummaryWriter(log_dir=log_dir, comment=comment)
        self.logger = logging.getLogger(f'[{self.method}]')
        self.z_metric = defaultdict(list)
        self.b_metric = defaultdict(list)
        self._train = True


    def get_train_pattern(self) -> torch.tensor:
        raise NotImplemented

    @torch.no_grad()
    def get_val_pattern(self) -> torch.tensor:
        raise NotImplemented

    def collect_grads_on_batch(self, f_func: Callable[[torch.tensor], torch.tensor]):
        z = self.get_train_pattern()
        f_func(z).backward()

    def refine_P_grads_on_batch(self):
        raise NotImplemented

    def refine_S_grads_on_batch(self):
        raise NotImplemented

    def update_on_batch(self, f_func: Callable[[torch.tensor], torch.tensor]):
        self.collect_grads_on_batch(f_func)
        self.refine_P_grads_on_batch()
        self.refine_S_grads_on_batch()

    @torch.no_grad()
    def get_pattern_image(self, pattern: torch.tensor) -> torch.tensor:
        if pattern.shape[0] == 1:
            img = pattern.cpu().detach().flatten().repeat(pattern.shape[-1], 1)
        else:
            img = pattern.cpu().detach()
        return img[None]

    @torch.no_grad()
    def update_on_val_batch(self,
                            epoch: int,
                            batch_idx: int,
                            scalar_funcs: Dict[str, Callable[[torch.tensor], torch.tensor]],
                            image_funcs: Dict[str, Callable[[torch.tensor], torch.tensor]],
                            verbose_image: bool
                            ):
        if self._train:
            self.z_metric = defaultdict(list)
            self.b_metric = defaultdict(list)
            self._train = False
        z = self.get_train_pattern()
        for func_name, func in scalar_funcs.items():
            self.z_metric[func_name].append(torch.nan_to_num(func(z), 0.0).item())
        b = self.get_val_pattern()
        for func_name, func in scalar_funcs.items():
            self.b_metric[func_name].append(torch.nan_to_num(func(b), 0.0).item())
        if verbose_image:
            z_img = self.get_pattern_image(z)
            b_img = self.get_pattern_image(b)
            self.writer.add_image(tag='pattern_z', img_tensor=z_img, global_step=epoch)
            self.writer.add_image(tag='pattern_b', img_tensor=b_img, global_step=epoch)
            for g_name, gfunc in image_funcs.items():
                yz = gfunc(z)
                yb = gfunc(b)
                if len(yz.shape) > 4:
                    yz = yz.movedim(1,2).flatten(0, 1)
                    yb = yb.movedim(1,2).flatten(0, 1)
                for c in range(yz.shape[1]):
                    yz_img = yz[:32, c][:, None].cpu().detach()
                    yb_img = yb[:32, c][:, None].cpu().detach()
                    self.writer.add_images(tag=f'{g_name}_z/{batch_idx}_{c}', img_tensor=t2i(yz_img), global_step=epoch)
                    self.writer.add_images(tag=f'{g_name}_b/{batch_idx}_{c}', img_tensor=t2i(yb_img), global_step=epoch)
        self.writer.flush()

    def update_on_train_end(self, epoch) -> None:
        pass

    def update_on_epoch(self, epoch) -> None:
        for metric, vals in self.z_metric.items():
            self.writer.add_scalar(tag=f'{metric}_z', scalar_value=np.mean(vals), global_step=epoch)
            self.writer.add_histogram(tag=f'stats_{metric}_z', values=np.array(vals), global_step=epoch)
        for metric, vals in self.b_metric.items():
            self.writer.add_scalar(tag=f'{metric}_b', scalar_value=np.mean(vals), global_step=epoch)
            self.writer.add_histogram(tag=f'stats_{metric}_b', values=np.array(vals), global_step=epoch)
        self.writer.flush()
        self.epoch = epoch
        self._train = False

    def update_on_end(self, save_path) -> None:
        hparams = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if isinstance(v, (int, float, str, bool)):
                    hparams[k] = v
                elif isinstance(v, (list, tuple)):
                    hparams[k] = 'x'.join([str(vv) for vv in v])
        metrics = {}
        for metric, vals in self.z_metric.items():
            metrics[f'{metric}_z'] = None
        for metric, vals in self.b_metric.items():
            metrics[f'{metric}_b'] = None
        self.writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)
        self.writer.flush()
        if save_path:
            state_vals = {}
            for k, v in self.__dict__.items():
                if not k.startswith('_'):
                    if isinstance(v, (int, float, str, bool, list, tuple, torch.Tensor, np.ndarray)):
                        state_vals[k] = v
            torch.save(state_vals, os.path.join(save_path, 'popt.pt_state'))
            b = self.get_val_pattern()
            torch.save(b.cpu(), os.path.join(save_path, 'pattern.pt'))
            if hasattr(self.model, 'net') and isinstance(self.model.net, torch.nn.ModuleList):
                torch.save(self.model.net.state_dict(), os.path.join(save_path, 'model_state.pt'))
            else:
                torch.save(self.model.state_dict(), os.path.join(save_path, 'model_state.pt'))


class GridOptimizer(PatternOptimizer):

    def __init__(
            self,
            model: torch.nn.Module,
            acceleration: float,
            img_shape: List[int],
            dimensions: int,
            device: torch.DeviceObjType,
            log_dir: str,
            comment: str,
    ) -> None:
        super(GridOptimizer, self).__init__(model, acceleration, img_shape, dimensions, device, log_dir, comment)

    def get_pattern_image(self, pattern: torch.tensor) -> torch.tensor:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(pattern[0].cpu().detach(), pattern[1].cpu().detach(), s=0.7)
        ax.set_xlim(-self.image_shape[0]//2, self.image_shape[0]//2)
        ax.set_ylim(-self.image_shape[1]//2, self.image_shape[1]//2)
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plt.close(fig)
        img_arr = cv2.imdecode(img_arr, 1)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        img_arr = torch.from_numpy(img_arr)
        return img_arr.permute(2, 0, 1)


class LOUPE(PatternOptimizer):

    def __init__(
            self,
            model: torch.nn.Module,
            acceleration: float,
            img_shape: List[int],
            dimensions: int,
            device: torch.DeviceObjType,
            log_dir: str,
            comment: str,
            lr: float,
            freeze_model: bool
    ) -> None:
        super().__init__(model, acceleration, img_shape, dimensions, device, log_dir, comment)
        self.theta = torch.nn.Parameter(
            data=self.__init_theta(self.image_shape, self.dimensions).to(self.device),
            requires_grad=True
        )
        self.lr = lr
        self.freeze_model = freeze_model
        if not freeze_model:
            self.model = self.model.train()
            for p in self.model.parameters():
                p.requires_grad = True
            self.optimizer = torch.optim.Adam([self.theta], lr=self.lr)
            self.optimizer_unet = torch.optim.Adam(
                self.model.parameters(),
                lr=3e-4 if 'IGS_S_LR' not in os.environ else float(os.environ['IGS_S_LR'])
            )
        else:
            self.optimizer = torch.optim.Adam([self.theta], lr=self.lr)

    @staticmethod
    def __norm_alpha(p, alpha):
        p_norm = p.norm(p=1) / reduce(lambda x, y: x*y, p.shape)
        if p_norm >= alpha:
            return alpha / p_norm * p
        else:
            return 1 - (1 - alpha) / (1 - p_norm) * (1 - p)

    @staticmethod
    def __init_theta(img_shape, dimensions):
        if dimensions == 1:
            return torch.cat([
                torch.linspace(0, 1, img_shape[1]//2),
                torch.linspace(1, 0, img_shape[1] - img_shape[1]//2)
            ])
        else:
            tx = torch.cat([
                torch.linspace(0, 1, img_shape[1]//2),
                torch.linspace(1, 0, img_shape[1] - img_shape[1]//2)
            ])
            ty = torch.cat([
                torch.linspace(0, 1, img_shape[0]//2),
                torch.linspace(1, 0, img_shape[0] - img_shape[0]//2)
            ])
            return ty[:, None] @ tx[None]

    def get_train_pattern(self) -> torch.tensor:
        z = (self.theta * 5).sigmoid()
        u = torch.rand_like(z)
        z = self.__norm_alpha(z, self.acceleration) - u
        z = (z * 200).sigmoid()
        if self.dimensions == 1:
            return z[None]
        else:
            return z

    @torch.no_grad()
    def get_val_pattern(self) -> torch.tensor:
        l0norm = int(self.acceleration * reduce(lambda x, y: x*y, self.theta.shape))
        b = torch.zeros_like(self.theta)
        b.flatten()[self.theta.flatten().argsort()[-l0norm:]] = 1
        if self.dimensions == 1:
            return b[None]
        else:
            return b

    def refine_P_grads_on_batch(self):
        torch.nn.utils.clip_grad_norm_([self.theta], 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def refine_S_grads_on_batch(self):
        if not self.freeze_model:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer_unet.step()
            self.optimizer_unet.zero_grad()

    def update_on_train_end(self, epoch) -> None:
        if not self.freeze_model:
            self.model = self.model.eval()
        super(LOUPE, self).update_on_train_end(epoch)

    def update_on_epoch(self, epoch) -> None:
        if not self.freeze_model:
            self.model = self.model.train()
        super(LOUPE, self).update_on_epoch(epoch)


class LOUPErc(LOUPE):

    def __init__(
            self,
            model: torch.nn.Module,
            acceleration: float,
            img_shape: List[int],
            dimensions: int,
            device: torch.DeviceObjType,
            log_dir: str,
            comment: str,
            lr: float,
            freeze_model: bool
    ) -> None:
        super(LOUPErc, self).__init__(model, acceleration, img_shape, dimensions, device, log_dir, comment,
                                      lr, freeze_model)

    @staticmethod
    def __norm_alpha(p, alpha):
        p_norm = p.norm(p=1) / reduce(lambda x,y: x*y, p.shape)
        m = torch.zeros_like(p)
        m[p >= alpha] = alpha / p_norm * p[p >= alpha]
        m[p < alpha] = 1 - (1 - alpha) / (1 - p_norm) * (1 - p[p < alpha])
        return m

    def get_train_pattern(self) -> torch.tensor:
        z = (self.theta * 5).sigmoid()
        u = torch.rand_like(z)
        z = self.__norm_alpha(z, self.acceleration) - u
        z = (z * 200).sigmoid()
        if self.dimensions == 1:
            return z[None]
        else:
            return z


class IGS(PatternOptimizer):

    def __init__(
            self,
            model: torch.nn.Module,
            acceleration: float,
            img_shape: List[int],
            dimensions: int,
            device: torch.DeviceObjType,
            log_dir: str,
            comment: str,
            lr: float,
            maxstep: int,
            freeze_model: bool
    ) -> None:
        super().__init__(model, acceleration, img_shape, dimensions, device, log_dir, comment)
        self.lr = lr
        self.pattern = torch.nn.Parameter(
            data=self.__init_w(self.image_shape, self.dimensions).to(self.device),
            requires_grad=True
        )
        self.pattern_history = list()
        self.grad = torch.zeros_like(self.pattern)
        self.maxstep = maxstep
        self.freeze_model = freeze_model
        if not freeze_model:
            self.model = self.model.train()
            for p in self.model.parameters():
                p.requires_grad = True
            self.optimizer = torch.optim.Adam([self.pattern], lr=self.lr)
            self.optimizer_unet = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam([self.pattern], lr=self.lr)


    @staticmethod
    def __init_w(img_shape, dimensions):
        if dimensions == 1:
            pattern = torch.zeros(img_shape[1])
            pattern[img_shape[1] // 2] = 1
            return pattern
        else:
            pattern = torch.zeros(*img_shape)
            pattern[img_shape[0]//2-1:img_shape[0]//2+1, img_shape[1]//2-1:img_shape[1]//2+1] = 1
            return pattern

    def get_train_pattern(self):
        z = self.pattern + torch.zeros_like(self.pattern)
        if self.dimensions == 1:
            return z[None]
        else:
            return z

    @torch.no_grad()
    def get_val_pattern(self):
        b = self.pattern + torch.zeros_like(self.pattern)
        if self.dimensions == 1:
            return b[None]
        else:
            return b

    def refine_P_grads_on_batch(self):
        pass

    def refine_S_grads_on_batch(self):
        pass

    def update_on_train_end(self, epoch):
        if not self.freeze_model:
            self.model = self.model.eval()
        z_grad = self.pattern.grad.flatten().data.clone()
        self.pattern.grad.zero_()
        self.optimizer.step()
        self.optimizer.zero_grad()
        grad_ind = z_grad.argsort()
        pattern_ind = torch.nonzero(self.pattern.flatten()[grad_ind] == 0).flatten()
        assert len(pattern_ind) > 0
        displ = min(self.maxstep, len(pattern_ind))
        self.pattern.data.flatten()[grad_ind[pattern_ind][:displ]] = 1
        self.pattern_history.append(self.pattern.clone().cpu().detach())
        super(IGS, self).update_on_train_end(epoch=epoch)

    def update_on_epoch(self, epoch) -> None:
        if not self.freeze_model:
            self.model = self.model.train()
        super(IGS, self).update_on_epoch(epoch)


class PILOT(GridOptimizer):

    def __init__(
            self,
            model: torch.nn.Module,
            acceleration: float,
            img_shape: List[int],
            dimensions: int,
            device: torch.DeviceObjType,
            log_dir: str,
            comment: str,
            lr: float,
            init_scale: int,
            freeze_model: bool,
            override_grid: torch.tensor
    ) -> None:
        super(PILOT, self).__init__(model, acceleration, img_shape, dimensions, device, log_dir, comment)
        self.init_scale = init_scale
        if dimensions == 1:
            self.grid_x = self.__init_x(img_shape, dimensions, acceleration, init_scale).to(self.device)
            self.grid_y = torch.nn.Parameter(
                data=self.__init_y(img_shape, dimensions, acceleration, init_scale).to(self.device)
                if override_grid is None else override_grid.to(self.device),
                requires_grad=True
            )
        else:
            grid_x = self.__init_x(img_shape, dimensions, acceleration, init_scale).to(self.device)
            grid_y = self.__init_y(img_shape, dimensions, acceleration, init_scale).to(self.device)
            self.grid = torch.nn.Parameter(
                data=torch.stack([arr.flatten() for arr in torch.meshgrid(grid_x, grid_y)])
                if override_grid is None else override_grid.to(self.device),
                requires_grad=True
            )
        self.lr = lr
        self.freeze_model = freeze_model
        if dimensions == 1:
            self.optimizer = torch.optim.Adam([self.grid_y], lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam([self.grid], lr=self.lr)
        if not freeze_model:
            self.model = self.model.train()
            for p in self.model.parameters():
                p.requires_grad = True
            self.optimizer_unet = torch.optim.Adam(
                self.model.parameters(),
                lr=3e-4 if 'IGS_S_LR' not in os.environ else float(os.environ['IGS_S_LR'])
            )


    @staticmethod
    def __init_x(image_shape, dimensions, acceleration, scale):
        if dimensions == 1:
            return torch.arange(-image_shape[0] // 2, image_shape[0] // 2).float()
        else:
            acc = acceleration ** 0.5
            return torch.arange(
                -round(image_shape[1] // 2 * acc) * scale,
                round(image_shape[1] // 2 * acc) * scale,
                step=scale
            ).float()

    @staticmethod
    def __init_y(image_shape, dimensions, acceleration, scale):
        if dimensions == 1:
            acc = acceleration
        else:
            acc = acceleration ** 0.5
        return torch.arange(
            -round(image_shape[1] // 2 * acc) * scale,
            round(image_shape[1] // 2 * acc) * scale,
            step=scale
        ).float()

    def get_train_pattern(self) -> torch.tensor:
        if self.dimensions == 1:
            return torch.stack([arr.flatten() for arr in torch.meshgrid(self.grid_x, self.grid_y)])
        else:
            return self.grid

    def get_val_pattern(self) -> torch.tensor:
        if self.dimensions == 1:
            return torch.stack([arr.flatten() for arr in torch.meshgrid(self.grid_x, self.grid_y)])
        else:
            return self.grid

    def refine_P_grads_on_batch(self):
        if self.dimensions == 1:
            torch.nn.utils.clip_grad_norm_([self.grid_y], 1.0)
        else:
            torch.nn.utils.clip_grad_norm_([self.grid], 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def refine_S_grads_on_batch(self):
        if not self.freeze_model:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer_unet.step()
            self.optimizer_unet.zero_grad()

    def update_on_train_end(self, epoch) -> None:
        if not self.freeze_model:
            self.model = self.model.eval()
        super(PILOT, self).update_on_train_end(epoch)

    def update_on_epoch(self, epoch) -> None:
        if not self.freeze_model:
            self.model = self.model.train()
        super(PILOT, self).update_on_epoch(epoch)


class MultiPatternOptimizer(PatternOptimizer):

    def __init__(
            self,
            pattern_optimizers: List[PatternOptimizer]
    ) -> None:
        self._po = pattern_optimizers
        _log_dir = pattern_optimizers[0].writer.log_dir
        for f in os.listdir(_log_dir):
            if f.startswith('events'):
                os.remove(os.path.join(_log_dir, f))
        super(MultiPatternOptimizer, self).__init__(pattern_optimizers[0].model.eval(),
                                                    pattern_optimizers[0].acceleration,
                                                    pattern_optimizers[0].image_shape,
                                                    pattern_optimizers[0].dimensions,
                                                    pattern_optimizers[0].device,
                                                    pattern_optimizers[0].log_dir,
                                                    pattern_optimizers[0].comment)
        self.freeze_model = pattern_optimizers[0].freeze_model if hasattr(pattern_optimizers[0], 'freeze_model') else False
        if not self.freeze_model:
            self.model = self.model.train()
            for p in self.model.parameters():
                p.requires_grad = True
            self.optimizer_unet = torch.optim.Adam(
                self.model.parameters(),
                lr=3e-4 if 'IGS_S_LR' not in os.environ else float(os.environ['IGS_S_LR'])
            )

    def get_train_pattern(self) -> torch.tensor:
        return torch.stack([p.get_train_pattern() for p in self._po])

    def get_val_pattern(self) -> torch.tensor:
        return torch.stack([p.get_val_pattern() for p in self._po])

    def collect_grads_on_batch(self, f_func: Callable[[torch.tensor], torch.tensor]):
        z = self.get_train_pattern()
        f_func(z).backward()

    def refine_P_grads_on_batch(self):
        for p in self._po:
            p.refine_P_grads_on_batch()

    def refine_S_grads_on_batch(self):
        if not self.freeze_model:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer_unet.step()
            self.optimizer_unet.zero_grad()

    def update_on_train_end(self, epoch) -> None:
        if not self.freeze_model:
            self.model = self.model.eval()
        for p in self._po:
            p.update_on_train_end(epoch)
        super(MultiPatternOptimizer, self).update_on_train_end(epoch)

    def update_on_epoch(self, epoch) -> None:
        if not self.freeze_model:
            self.model = self.model.train()
        for p in self._po:
            p.update_on_epoch(epoch)
        super(MultiPatternOptimizer, self).update_on_epoch(epoch)

    @torch.no_grad()
    def update_on_val_batch(self,
                            epoch: int,
                            batch_idx: int,
                            scalar_funcs: Dict[str, Callable[[torch.tensor], torch.tensor]],
                            image_funcs: Dict[str, Callable[[torch.tensor], torch.tensor]],
                            verbose_image: bool
                            ):
        if self._train:
            self.z_metric = defaultdict(list)
            self.b_metric = defaultdict(list)
            self._train = False
        z = self.get_train_pattern()
        for func_name, func in scalar_funcs.items():
            self.z_metric[func_name].append(torch.nan_to_num(func(z), 0.0).item())
        b = self.get_val_pattern()
        for func_name, func in scalar_funcs.items():
            self.b_metric[func_name].append(torch.nan_to_num(func(b), 0.0).item())
        if verbose_image:
            for i, p in enumerate(self._po):
                z = p.get_train_pattern()
                b = p.get_val_pattern()
                z_img = self.get_pattern_image(z)
                b_img = self.get_pattern_image(b)
                self.writer.add_image(tag=f'pattern_z_{i}', img_tensor=z_img, global_step=epoch)
                self.writer.add_image(tag=f'pattern_b_{i}', img_tensor=b_img, global_step=epoch)
            for g_name, gfunc in image_funcs.items():
                yz = gfunc(z)
                yb = gfunc(b)
                if len(yz.shape) > 4:
                    yz = yz.movedim(1,2).flatten(0, 1)
                    yb = yb.movedim(1,2).flatten(0, 1)
                for c in range(yz.shape[1]):
                    yz_img = yz[:32, c][:, None].cpu().detach()
                    yb_img = yb[:32, c][:, None].cpu().detach()
                    self.writer.add_images(tag=f'{g_name}_z/{batch_idx}_{c}', img_tensor=t2i(yz_img), global_step=epoch)
                    self.writer.add_images(tag=f'{g_name}_b/{batch_idx}_{c}', img_tensor=t2i(yb_img), global_step=epoch)
        self.writer.flush()


class MetricFuncGen:

    def __init__(self, name):
        self.name = name

    def __call__(self,
                 y: Dict[str, torch.tensor],
                 x: Dict[str, torch.tensor],
                 s_func: torch.nn.Module):
        if self.name == 'dice':
            return dice_loss(y['mask'].long(), s_func(y['img']))
        elif self.name == 'dice_3d':
            return dice_loss(y['mask'].long().movedim(1,2).flatten(0, 1), s_func(y['img']).movedim(1,2).flatten(0, 1))
        elif self.name == 'dice_322d':
            return dice_loss(y['mask'].long().flatten(0, 1).movedim(0,1), s_func(y['img'].flatten(0, 1).movedim(0,1)))
        elif self.name == 'ce':
            t = y['mask'].long().movedim(1,2).flatten(0, 1)
            p = s_func(y['img']).movedim(1,2).flatten(0, 1)
            return ce_loss(t.squeeze(1), p, weights=None)
        elif self.name == 'ce_3d':
            t = y['mask'].long()
            p = s_func(y['img'])
            return ce_loss(t.squeeze(1), p, weights=None)
        elif self.name == 'dice_vec':
            return dice_coeffs(y['mask'].long(), s_func(y['img']))
        elif self.name == 'dice_3d_acdc_vec':
            return dice_3d_acdc_vec(y['mask'].long().flatten(0, 1).movedim(0,1), s_func(y['img'].flatten(0, 1).movedim(0,1)))
        elif self.name == 'dice_3d_acdc_3d_vec':
            return dice_3d_acdc_vec(y['mask'].long().movedim(1,2).flatten(0, 1), s_func(y['img']).movedim(1,2).flatten(0, 1))
        elif self.name == 'dice_3d_brats_vec':
            return dice_3d_brats_vec(y['mask'].long().flatten(0, 1).movedim(0,1), s_func(y['img'].flatten(0, 1).movedim(0,1)))
        elif self.name == 'dice_3d_brats_3d_vec':
            return dice_3d_brats_vec(y['mask'].long().movedim(1,2).flatten(0, 1), s_func(y['img']).movedim(1,2).flatten(0, 1))
        elif self.name == 'l1':
            return F.l1_loss(y['img'], x['img'])
        elif self.name == 'ssim':
            # return -pt_ssim2(y['img'] * y['std'] + y['mean'], x['img'] * x['std'] + x['mean'], val_range=(x['img'] * x['std'] + x['mean']).max())
            return -pt_ssim(y['img'], x['img'])
        elif self.name == 'ssim2':
            return -pt_ssim2(y['img'] * y['std'] + y['mean'], x['img'] * x['std'] + x['mean'],
                             val_range=(x['img'] * x['std'] + x['mean']).max())
        elif self.name == 'ssim_322d':
            return -pt_ssim(y['img'].flatten(0, 1).movedim(0,1), x['img'].flatten(0, 1).movedim(0,1))
        elif self.name == 'ssim_3d':
            return -pt_ssim(y['img'].movedim(1,2).flatten(0, 1),
                            x['img'].movedim(1,2).flatten(0, 1))
        elif self.name == 'dice_ce':
            t = y['mask'].long()
            p = s_func(y['img'])
            return dice_loss(t, p) * .75 + ce_loss(t.squeeze(1), p, weights=None) * .25
        elif self.name == 'dice_ce_3d':
            t = y['mask'].long().movedim(1,2).flatten(0, 1)
            p = s_func(y['img']).movedim(1,2).flatten(0, 1)
            return dice_loss(t, p) * .75 + ce_loss(t.squeeze(1), p, weights=None) * .25
        elif self.name == 'cs-ssim':
            return -pt_ssim(s_func.forward(y), x['img'])
        elif self.name == 'cs-ssim2':
            p = s_func.forward(y)
            return -pt_ssim2(p * y['std'] + y['mean'], x['img'] * x['std'] + x['mean'],
                             val_range=(x['img'] * x['std'] + x['mean']).max())
        elif self.name == 'cs-ssim-l1':
            p = s_func.forward(y)
            t = x['img']
            return (1 - 0.84) * F.l1_loss(p, t) + 0.84 * (1 - pt_ssim(p, t))
        elif self.name == 'cs-ssim2-l1':
            p = s_func.forward(y)
            t = x['img']
            return (1 - 0.84) * F.l1_loss(p, t) + 0.84 * (1 - pt_ssim2(p * y['std'] + y['mean'], x['img'] * x['std'] + x['mean'], val_range=(x['img'] * x['std'] + x['mean']).max()))
        elif self.name == 'cs-l1':
            return F.l1_loss(s_func.forward(y), x['img'])
        elif self.name == 'cs-plus-ssim':
            p = s_func.forward(y)
            t = x['img']
            bboxes = x['boxes']
            res = 0
            count = 0
            for i, boxes in enumerate(bboxes):
                for (y0, x0, y1, x1) in boxes:
                    res -= pt_ssim(p[i,:,x0:x1,y0:y1][None], t[i,:,x0:x1,y0:y1][None])
                    count += 1
            return (res / count) * 0.8 - pt_ssim(p, t) * 0.2
        elif self.name == 'cs-plus-ssim2':
            p = s_func.forward(y)
            t = x['img']
            bboxes = x['boxes']
            res = 0
            count = 0
            for i, boxes in enumerate(bboxes):
                for (y0, x0, y1, x1) in boxes:
                    pp = (p * y['std'] + y['mean'])[i,:,x0:x1,y0:y1][None]
                    tt = (t * x['std'] + x['mean'])[i,:,x0:x1,y0:y1][None]
                    res -= pt_ssim2(pp, tt, val_range=tt.max())
                    count += 1
            return (res / count) * 0.8 - pt_ssim2(p * y['std'] + y['mean'], t * x['std'] + x['mean'], val_range=(t * x['std'] + x['mean']).max()) * 0.2
        elif self.name == 'cs-plus-l1':
            p = s_func.forward(y)
            t = x['img']
            bboxes = x['boxes']
            res = 0
            count = 0
            for i, boxes in enumerate(bboxes):
                for (y0, x0, y1, x1) in boxes:
                    res += F.l1_loss(p[i,:,x0:x1,y0:y1][None], t[i,:,x0:x1,y0:y1][None])
                    count += 1
            return (res / count) * 0.8 + F.l1_loss(p, t) * 0.2
        elif self.name == 'cs-plus-ssim-l1':
            p = s_func.forward(y)
            t = x['img']
            bboxes = x['boxes']
            res = 0
            count = 0
            for i, boxes in enumerate(bboxes):
                for (y0, x0, y1, x1) in boxes:
                    res += (1 - 0.84) * F.l1_loss(p[i,:,x0:x1,y0:y1][None], t[i,:,x0:x1,y0:y1][None]) \
                           + 0.84 * (1 - pt_ssim(p[i,:,x0:x1,y0:y1][None], t[i,:,x0:x1,y0:y1][None]))
                    count += 1
            return (res / count) * 0.8 + ((1 - 0.84) * F.l1_loss(p, t) + 0.84 * (1 - pt_ssim(p, t))) * 0.2
        elif self.name == 'cs-plus-ssim2-l1':
            p = s_func.forward(y)
            t = x['img']
            bboxes = x['boxes']
            res = 0
            count = 0
            for i, boxes in enumerate(bboxes):
                for (y0, x0, y1, x1) in boxes:
                    pp = (p * y['std'] + y['mean'])[i,:,x0:x1,y0:y1][None]
                    tt = (t * x['std'] + x['mean'])[i,:,x0:x1,y0:y1][None]
                    res += (1 - 0.84) * F.l1_loss(p[i,:,x0:x1,y0:y1][None], t[i,:,x0:x1,y0:y1][None]) \
                           + 0.84 * (1 - pt_ssim2(pp, tt, val_range=tt.max()))
                    count += 1
            return (res / count) * 0.8 + ((1 - 0.84) * F.l1_loss(p, t) + 0.84 * (1 - pt_ssim2(p * y['std'] + y['mean'], t * x['std'] + x['mean'], val_range=(t * x['std'] + x['mean']).max()))) * 0.2
        else:
            raise ValueError


class _FfuncGen:

    def __init__(self, data, model, sampler, func):
        self.data = data
        self.model = model
        self.sampler = sampler
        self.func = func

    def __call__(self, p):
        return self.func(y=self.sampler(self.data, sampling=p), x=self.data, s_func=self.model)


class ACDCDataset(torch.utils.data.Dataset):
    CLASSES = {0: 'NOR', 1: 'MINF', 2: 'DCM', 3: 'HCM', 4: 'RV'}

    def __init__(self, hf_path, device):
        super().__init__()
        self.device = device
        self.hf = h5py.File(hf_path, mode='r')

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, item: int):
        img = self.hf[str(item)][:1]
        mask = self.hf[str(item)][1:]
        c = self.hf[str(item)].attrs['class']
        img = torch.tensor(img).float()
        mask = torch.tensor(mask)
        mean = img.mean()
        std = img.std()
        img = (img - mean) / (std + 1e-11)
        return dict(
            c=c,
            mask=mask.to(self.device),
            img=img.to(self.device),
            mean=mean[None, None, None].to(self.device),
            std=std[None, None, None].to(self.device)
        )


class BraTS2dDataset(torch.utils.data.Dataset):

    def __init__(self, hf_path, device):
        super().__init__()
        self.hf = h5py.File(hf_path, mode='r')
        self.device = device

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, item: int):
        img = self.hf[str(item)][:-1,:,:]
        mask = self.hf[str(item)][-1:,:,:]
        if 'IGS_BRATS_T' in os.environ and os.environ['IGS_BRATS_T'] == '1':
            img = torch.tensor(img).float()
            mask = torch.tensor(mask).long()
        else:
            img = torch.tensor(img).float().movedim(-1, -2)
            mask = torch.tensor(mask).long().movedim(-1, -2)
        mean = img.mean()
        std = img.std()
        img = (img - mean) / (std + 1e-11) + 1e-11
        return dict(
            mask=mask.to(self.device),
            img=img.to(self.device),
            mean=mean[None, None, None].to(self.device),
            std=std[None, None, None].to(self.device)
        )


class ACDCDataset3D(torch.utils.data.Dataset):
    CLASSES = {0: 'NOR', 1: 'MINF', 2: 'DCM', 3: 'HCM', 4: 'RV'}

    def __init__(self, hf_path, device):
        super().__init__()
        self.device = device
        self.hf = h5py.File(hf_path)
        self.vols = []
        for k in self.hf.keys():
            for kk in self.hf[k].keys():
                self.vols.append((k, kk))

    def __len__(self) -> int:
        return len(self.vols)

    def __getitem__(self, item: int):
        k, kk = self.vols[item]
        img = self.hf[k][kk][:1]
        mask = self.hf[k][kk][1:]

        img = torch.tensor(img).float()
        mask = torch.tensor(mask)
        mean = img.mean(dim=(2,3))[:, :, None,None]
        std = img.std(dim=(2,3))[:, :, None,None]
        img = (img - mean) / (std + 1e-11)
        return dict(
            mask=mask.to(self.device),
            img=img.to(self.device),
            mean=mean.to(self.device),
            std=std.to(self.device)
        )

class BraTS3dDataset(torch.utils.data.Dataset):

    def __init__(self, hf_path, device):
        super().__init__()
        self.hf = h5py.File(hf_path, mode='r')
        self.device = device
        self.indexes = [k for k in self.hf.keys()]

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, item: int):
        key = self.indexes[item]
        img = self.hf[key][:-1, :, :, :]
        mask = self.hf[key][-1:, :, :, :]
        img = torch.tensor(img).float()
        mask = torch.tensor(mask).long()
        if 'IGS_BRATS_T' in os.environ and os.environ['IGS_BRATS_T'] == '1':
            img = img.movedim(3, 1)
            mask = mask.movedim(3, 1)
        else:
            img = img.movedim(3, 1).movedim(-1, -2)
            mask = mask.movedim(3, 1).movedim(-1, -2)
        mean = img.mean(dim=(2, 3))[:, :, None,None]
        std = img.std(dim=(2, 3))[:, :, None,None]
        img = (img - mean) / (std + 1e-11)
        return dict(
            mask=mask.to(self.device),
            img=img.to(self.device),
            mean=mean.to(self.device),
            std=std.to(self.device)
        )


class FastMRIDataset2D(torch.utils.data.Dataset):

    def __init__(self, hf_path, device):
        super().__init__()
        self.hf = h5py.File(hf_path, mode='r')
        self.slices = []
        for k in self.hf.keys():
            for i in range(self.hf[k].shape[0]):
                self.slices.append((k, i))
        self.device = device

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, item):
        key, slice = self.slices[item]
        ks = self.hf[key][slice] * 1e6
        xs = (ks.shape[-2] - 640) // 2
        ys = (ks.shape[-1] - 320) // 2
        xt = xs + 640
        yt = ys + 320
        ks = torch.tensor(ks).cfloat()[xs:xt, ys:yt][None].to(self.device)
        img = IFt(ks).abs()
        if 'IGS_FASTMRI_S' in os.environ and os.environ['IGS_FASTMRI_S'] == '1':
            img = img[:, 160:-160, :]
        mean = img.mean()
        std = img.std()
        img = (img - mean) / (std + 1e-11)
        return dict(
            img=img,
            mean=mean[None, None, None].to(self.device),
            std=std[None, None, None].to(self.device)
        )


class FastMRIDataset3D(torch.utils.data.Dataset):

    def __init__(self, hf_path, device):
        super().__init__()
        self.hf = h5py.File(hf_path, mode='r')
        self.vols = list(self.hf.keys())
        self.device = device

    def __len__(self):
        return len(self.vols)

    def __getitem__(self, item):
        key = self.vols[item]
        ks = self.hf[key][:] * 1e6
        xs = (ks.shape[-2] - 640) // 2
        ys = (ks.shape[-1] - 320) // 2
        xt = xs + 640
        yt = ys + 320
        ks = torch.tensor(ks).cfloat()[:, xs:xt, ys:yt][None].to(self.device)
        img = IFt(ks).abs()
        if 'IGS_FASTMRI_S' in os.environ and os.environ['IGS_FASTMRI_S'] == '1':
            img = img[:, :, 160:-160, :]
        mean = img.mean(dim=(2,3))[:, :, None,None]
        std = img.std(dim=(2,3))[:, :, None,None]
        img = (img - mean) / (std + 1e-11)
        return dict(
            img=img.to(self.device),
            mean=mean.to(self.device),
            std=std.to(self.device)
        )


class FastMRIplusDataset2D(torch.utils.data.Dataset):

    def __init__(self, hf_path: str, csv_path: str, ignore_slices: Set[int],
                 skip_empty: bool, skip_empty_slices: bool, device):
        super().__init__()
        self.hf = h5py.File(hf_path, 'r')
        self.df = pd.read_csv(csv_path, index_col=None, header=0, sep=',')
        self.classes = [(i,c) for i,c in enumerate(set(self.df.label.tolist()).difference({'artifact'}))]
        self.cat2label = {i:c for (i,c) in self.classes}
        self.label2cat = {c:i for (i,c) in self.classes}
        self.ignore_files = set(self.df[self.df.label == 'artifact'].file.tolist())
        self.ignore_slices = ignore_slices
        self.slices = []
        self.min_box_size = int(os.environ['IGS_FASTMRI_PLUS_MIN_BOX_SIZE']) if 'IGS_FASTMRI_PLUS_MIN_BOX_SIZE' in os.environ else 12
        fnames = set(self.df.file.tolist())
        for k in self.hf.keys():
            fname = k.split('.')[0]
            if fname not in self.ignore_files:
                if (not skip_empty) or (skip_empty and fname in fnames):
                    for i in range(self.hf[k].shape[0]):
                        if i not in ignore_slices:
                            if (not skip_empty_slices) or (skip_empty_slices and len(self.df[(self.df.file == fname) & (self.df.slice == i)])):
                                self.slices.append((k, i))
        self.device = device

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, item):
        key, z = self.slices[item]
        ks = torch.tensor(self.hf[key][z], dtype=torch.cfloat).flip(0) * 1e6
        h, w = ks.shape
        if h == 320 and w == 320:
            img = IFt(ks).abs()
        else:
            img = IFt(ks)[h//2-160:h//2+160, w//2-160:w//2+160].abs()
        mean = img.mean()
        std = img.std()
        img = (img - mean) / (std + 1e-11)
        labels_for_file = self.df.loc[self.df['file'] == key.split('.')[0]]
        labels_for_file['label'].unique()
        labels_for_slice = labels_for_file.loc[labels_for_file['slice'] == z].values.tolist()
        boxes = []
        labels = []
        for box in labels_for_slice:
            _, _, _, x0, y0, w, h, label = box
            x0, y0, x1, y1 = x0, y0, x0 + w + 1, y0 + h + 1
            xc = (x1 + x0) // 2 + 1
            yc = (y1 + y0) // 2 + 1
            ww = max(w // 2 + 1, self.min_box_size)
            hh = max(h // 2 + 1, self.min_box_size)
            x0, y0, x1, y1 = xc - ww, yc - hh, xc + ww, yc + hh
            boxes.append([int(x0), int(y0), int(x1), int(y1)])
            labels.append(self.label2cat[label])
        return dict(
            img=img[None].to(self.device),
            mean=mean[None, None, None].to(self.device),
            std=std[None, None, None].to(self.device),
            boxes=boxes,
        )


class DatasetOnDevice(torch.utils.data.Dataset):

    def __init__(self, dataset, device):
        super().__init__()
        self.memory = defaultdict(list)
        for i in tqdm(range(len(dataset)), desc=f'load {dataset} to {device}'):
            data = dataset[i]
            for k in data:
                self.memory[k].append(
                    data[k].to(device) if isinstance(data[k], torch.Tensor) else torch.tensor(data[k]).to(device))

    def __len__(self):
        keys = list(self.memory.keys())
        return len(self.memory[keys[0]])

    def __getitem__(self, index):
        return {k: self.memory[k][index] for k in self.memory.keys()}


def get_dataset(name, device, volumetric=False, preload=False):
    if name == 'acdc':
        if volumetric:
            train_dataset = ACDCDataset3D(ACDC_3D_TRAIN, device)
            val_dataset = ACDCDataset3D(ACDC_3D_VAL, device)
        else:
            train_dataset = ACDCDataset(ACDC_2D_TRAIN, device)
            val_dataset = ACDCDataset(ACDC_2D_VAL, device)
    elif name == 'brats':
        if volumetric:
            train_dataset = BraTS3dDataset(BRATS_3D_TRAIN, device)
            val_dataset = BraTS3dDataset(BRATS_3D_VAL, device)
        else:
            train_dataset = BraTS2dDataset(BRATS_2D_TRAIN, device)
            val_dataset = BraTS2dDataset(BRATS_2D_VAL, device)
    elif name == 'fastmri':
        if 'IGS_FASTMRI_FULL' in os.environ and os.environ['IGS_FASTMRI_FULL'] == '1':
            if volumetric:
                train_dataset = FastMRIDataset3D(FASTMRI_FULL_3D_TRAIN, device)
                val_dataset = FastMRIDataset3D(FASTMRI_FULL_3D_VAL, device)
            else:
                train_dataset = FastMRIDataset2D(FASTMRI_FULL_2D_TRAIN, device)
                val_dataset = FastMRIDataset2D(FASTMRI_FULL_2D_VAL, device)
        else:
            if volumetric:
                train_dataset = FastMRIDataset3D(FASTMRI_3D_TRAIN, device)
                val_dataset = FastMRIDataset3D(FASTMRI_3D_VAL, device)
            else:
                train_dataset = FastMRIDataset2D(FASTMRI_2D_TRAIN, device)
                val_dataset = FastMRIDataset2D(FASTMRI_2D_VAL, device)
    elif name == 'fastmri_plus':
        if 'IGS_FASTMRI_FULL' in os.environ and os.environ['IGS_FASTMRI_FULL'] == '1':
            if volumetric:
                raise ValueError
            else:
                train_dataset = FastMRIplusDataset2D(
                    hf_path=FASTMRI_FULL_2D_TRAIN,
                    csv_path='annotations/knee.csv',
                    ignore_slices=set(),
                    skip_empty=True,
                    skip_empty_slices=True,
                    device=device
                )
                val_dataset = FastMRIplusDataset2D(
                    hf_path=FASTMRI_FULL_2D_VAL,
                    csv_path='annotations/knee.csv',
                    ignore_slices=set(),
                    skip_empty=True,
                    skip_empty_slices=True,
                    device=device
                )
        else:
            if volumetric:
                raise ValueError
            else:
                train_dataset = FastMRIplusDataset2D(
                    hf_path=FASTMRI_2D_TRAIN,
                    csv_path='annotations/knee.csv',
                    ignore_slices=set(),
                    skip_empty=True,
                    skip_empty_slices=True,
                    device=device
                )
                val_dataset = FastMRIplusDataset2D(
                    hf_path=FASTMRI_2D_VAL,
                    csv_path='annotations/knee.csv',
                    ignore_slices=set(),
                    skip_empty=True,
                    skip_empty_slices=True,
                    device=device
                )
    else:
        raise ValueError
    # train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(20).numpy().tolist())
    # val_dataset = torch.utils.data.Subset(val_dataset, torch.arange(12).numpy().tolist())
    if preload:
        return DatasetOnDevice(train_dataset, device), DatasetOnDevice(val_dataset, device)
    else:
        return train_dataset, val_dataset


def get_segmentation_model(dataset_name, name, f_path, nchans, nlayers, device):
    if name == 'unet':
        model = Unet(IMG_CHANNELS[dataset_name], 4, chans=nchans, num_pool_layers=nlayers).to(device).train(False).eval()
        for p in model.parameters():
            p.requires_grad = False
        model.load_state_dict(torch.load(f_path, map_location=device))
        return model
    elif name == 'unet-att':
        model = AttU_Net(IMG_CHANNELS[dataset_name], 4, nchans).to(device).train(False).eval() # 32 channels
        for p in model.parameters():
            p.requires_grad = False
        model.load_state_dict(torch.load(f_path, map_location=device))
        return model
    elif name == 'unet3d':
        model = UNet3d(IMG_CHANNELS[dataset_name], 4, nchans).to(device).train(False).eval() # 24 channels
        for p in model.parameters():
            p.requires_grad = False
        model.load_state_dict(torch.load(f_path, map_location=device))
        return model
    elif name == 'none':
        return torch.nn.Identity()
    elif name == 'cs-unet':
        model = UnetCS(in_chans=IMG_CHANNELS[dataset_name], out_chans=1, chans=nchans, num_pool_layers=nlayers).to(device).train(False).eval()
        for p in model.parameters():
            p.requires_grad = False
        model.load_state_dict(torch.load(f_path, map_location=device))
        return model
    elif name == 'cs-cascade':
        model = CascadeCS(chans=nchans, num_pool_layers=nlayers).to(device).train(False).eval()
        for p in model.parameters():
            p.requires_grad = False
        model.net.load_state_dict(torch.load(f_path, map_location=device))
        return model
    else:
        raise ValueError


def _get_pattern_optimizer(
        model, freeze_model, mode, method, acceleration, imsize, device,
        log_dir, comment, maxstep, lr, init_scale, pattern, img_channels):
    if mode == '1d':
        dimensions = 1
    elif mode == '2d':
        dimensions = 2
    else:
        raise ValueError
    if method == 'loupe':
        return [LOUPE(
            model=model,
            acceleration=1 / acceleration,
            img_shape=imsize,
            dimensions=dimensions,
            device=device,
            log_dir=log_dir,
            comment=comment,
            lr=lr,
            freeze_model=freeze_model
        ) for _ in range(img_channels)]
    elif method == 'loupe_rc':
        return [LOUPErc(
            model=model,
            acceleration=1 / acceleration,
            img_shape=imsize,
            dimensions=dimensions,
            device=device,
            log_dir=log_dir,
            comment=comment,
            lr=lr,
            freeze_model=freeze_model
        ) for _ in range(img_channels)]
    elif method == 'igs':
        return [IGS(
            model=model,
            acceleration=1 / acceleration,
            img_shape=imsize,
            dimensions=dimensions,
            device=device,
            maxstep=maxstep,
            log_dir=log_dir,
            comment=comment,
            lr=lr,
            freeze_model=freeze_model
        ) for _ in range(img_channels)]
    elif method == 'pilot':
        override_grid = None
        if pattern:
            p = torch.load(pattern)
            mask = torch.ones(imsize) * p
            grid_x = torch.arange(-imsize[0] // 2, imsize[0] // 2).float()
            grid_y = torch.arange(-imsize[1] // 2, imsize[1] // 2).float()
            if dimensions == 1:
                override_grid = grid_y[p.flatten() == 1]
            else:
                grid = torch.stack([arr.flatten() for arr in torch.meshgrid(grid_x, grid_y)])
                override_grid = grid[:, mask.flatten() == 1]
        return [PILOT(
            model=model,
            acceleration=1 / acceleration,
            img_shape=imsize,
            dimensions=dimensions,
            device=device,
            log_dir=log_dir,
            comment=comment,
            lr=lr,
            init_scale=init_scale,
            freeze_model=freeze_model,
            override_grid=override_grid
        ) for _ in range(img_channels)]
    else:
        raise ValueError

def get_pattern_optimizer(
        model, freeze_model, mode, method, acceleration, imsize, device,
        log_dir, comment, maxstep, lr, init_scale, pattern, img_channels):
    patterns = _get_pattern_optimizer(model, freeze_model, mode, method, acceleration, imsize, device, log_dir, comment, maxstep, lr, init_scale, pattern, img_channels)
    if img_channels == 1:
        return patterns[0]
    else:
        return MultiPatternOptimizer(patterns)

def set_logging(args):
    log_dir = os.path.join(
        args.log_dir,
        args.dataset,
        args.model,
        args.method,
        '_'.join([f'x{args.acceleration}', f'{"multi" if args.multicoil else "single"}coil', args.mode, args.loss])
    )
    version = 1
    while os.path.exists(os.path.join(log_dir, f'v{version}')):
        version += 1
    log_dir = os.path.join(log_dir, f'v{version}')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(
            log_dir,
            f'{args.method}_'
            f'{args.acceleration}_'
            f'{"multi" if args.multicoil else "single"}coil_'
            f'{args.dataset}_model{"Fixed" if args.freeze_model else "Tuned"}'
            f'{args.mode}.log'),
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    d_keys = list(os.environ.keys())
    d_keys = list(filter(lambda s: s.startswith('IGS'), d_keys))
    logging.info('Environ:\n\t' + '\n\t'.join([f'{k}={os.environ[k]}' for k in d_keys]))
    logging.info('Params:\n\t' + '\n\t'.join([f'{k} : {v}' for k,v in args.__dict__.items()]))
    return log_dir


def get_args(parser):
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs/pattern'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['unet', 'unet-att', 'unet3d', 'none', 'cs-unet', 'cs-cascade']
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--freeze_model',
        default=True,
        type=lambda x: (str(x).lower() == 'true'),
    )
    parser.add_argument(
        '--pattern',
        default=None,
        type=str,
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
        '--comment',
        type=str,
        default='unnamed'
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
        '--dataset',
        type=str,
        choices=['acdc', 'brats', 'fastmri', 'fastmri_plus'],
        required=True
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
        '--r',
        default=0,
        type=int,
        choices=[0,1,2,3,4],
        help='0 - train and validate each epoch, 1 - only train, 2 - train and validate on 5-fold CV'
    )
    parser.add_argument(
        '--method',
        choices=['loupe', 'loupe_rc', 'igs', 'pilot', 'center', 'equispaced'],
        required=True
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['1d', '2d'],
        required=True
    )
    parser.add_argument(
        '--batch',
        default=32,
        type=int,
    )
    parser.add_argument(
        '--val_batch',
        default=8,
        type=int,
    )
    parser.add_argument(
        '--acceleration',
        default=16,
        type=int,
        required=True
    )
    parser.add_argument(
        '--epoch',
        default=100,
        type=int,
    )
    parser.add_argument(
        '--maxstep',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--lr',
        default=1e-2,
        type=float,
    )
    parser.add_argument(
        '--init_scale',
        default=1,
        type=float,
    )
    parser.add_argument(
        '--val_log_step',
        default=1,
        type=float,
    )
    return parser.parse_args()


COIL_MAP = dict(acdc='smap_256.pt', brats='smap_240.pt')
IMG_CHANNELS = dict(acdc=1, brats=4, fastmri=1, fastmri_plus=1)


def refine_data_batch(data):
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        boxes = [d['boxes'] for d in data]
        img = torch.stack([d['img'] for d in data])
        mean = torch.stack([d['mean'] for d in data])
        std = torch.stack([d['std'] for d in data])
        sampling = data[0]['sampling'] if 'sampling' in data[0] else None
        mask = torch.stack([d['mask'] for d in data]) if 'mask' in data[0] else None
        return dict(mask=mask, img=img, mean=mean, std=std, sampling=sampling, boxes=boxes)
    else:
        raise ValueError


def train(args, size_x, size_y, device, log_dir, k_sampler, train_generator, val_generator):
    # Load segmentation model and set pattern optimizer
    p_optimizer = get_pattern_optimizer(
        model=get_segmentation_model(args.dataset, args.model, f_path=args.model_path, nchans=args.nchans, nlayers=args.nlayers,
                                     device=device),
        freeze_model=args.freeze_model, mode=args.mode, method=args.method, acceleration=args.acceleration,
        imsize=[size_x, size_y], device=device, log_dir=log_dir, comment=args.comment, maxstep=args.maxstep, lr=args.lr,
        init_scale=args.init_scale, pattern=args.pattern, img_channels=IMG_CHANNELS[args.dataset])
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
    # Set image verbose freq
    _verbose_train_freq = int(len(train_generator) * 0.5) + 1
    _verbose_val_freq = int(len(val_generator) * 0.33) + 1
    # Set iter num
    if args.method == 'igs':
        # TODO: do for IGS
        niter = int(size_x * size_y / args.acceleration) if args.mode == '2d' else int(size_y / args.acceleration)
        niter -= 4 if args.mode == '2d' else 1
        niter = int(niter / args.maxstep)
    else:
        niter = args.epoch
    # Run iterations
    if args.r == 0:
        total_steps = niter * (len(train_generator) + len(val_generator))
    else:
        total_steps = niter * len(train_generator)
    pbar = tqdm(total=total_steps)
    for epoch in range(niter):
        try:
            for i, data in enumerate(train_generator):
                data = refine_data_batch(data)
                p_optimizer.update_on_batch(f_func=lambda p: loss_func(y=k_sampler(data, sampling=p),
                                                                       x=data,
                                                                       s_func=p_optimizer.model))
                pbar.set_description(f'training: [{epoch} epoch] [{i}/{len(train_generator)}]')
                pbar.update(1)
                if i % _verbose_train_freq == 0:
                    logging.info(f'[Training {epoch} epoch]: batch [{i}/{len(train_generator)}]')
            p_optimizer.update_on_train_end(epoch)
            if args.r == 0:
                for i, data in enumerate(val_generator):
                    data = refine_data_batch(data)
                    if (i % _verbose_val_freq == 0) and (epoch % args.val_log_step == 0):
                        p_optimizer.update_on_val_batch(epoch=epoch, batch_idx=i,
                                                        scalar_funcs={
                                                            metric: _FfuncGen(data, p_optimizer.model, k_sampler, f)
                                                            for metric, f in metric_funcs.items()
                                                        },
                                                        image_funcs=dict(image=lambda x: k_sampler(data, sampling=x)['img']),
                                                        verbose_image=True)
                    else:
                        p_optimizer.update_on_val_batch(epoch=epoch, batch_idx=i,
                                                        scalar_funcs={
                                                            metric: _FfuncGen(data, p_optimizer.model, k_sampler, f)
                                                            for metric, f in metric_funcs.items()
                                                        },
                                                        image_funcs=dict(image=lambda x: k_sampler(data, sampling=x)['img']),
                                                        verbose_image=False)
                    pbar.set_description(f'validation: [{epoch} epoch] [{i}/{len(val_generator)}]')
                    pbar.update(1)
                    if i % _verbose_val_freq == 0:
                        logging.info(f'[Validation {epoch} epoch]: batch [{i}/{len(val_generator)}]')
            p_optimizer.update_on_epoch(epoch)
        except KeyboardInterrupt:
            print('Try stopping...')
            break
    pbar.close()
    p_optimizer.update_on_end(log_dir)
    return p_optimizer


def val(args, pattern, model, k_sampler, val_generator, csv_fp):
    # TODO: it do not work and not so much needed, but still need fix!
    model.eval()
    results = defaultdict(list)
    if args.dataset != 'fastmri' and args.model != 'none':
        if args.model != 'unet3d':
            metric_funcs = dict(dice=MetricFuncGen('dice'),
                                dice_vec=MetricFuncGen('dice_vec'),
                                ssim=MetricFuncGen('ssim'))
        else:
            metric_funcs = dict(dice=MetricFuncGen('dice_3d'),
                                ssim=MetricFuncGen('ssim_3d'))
    else:
        metric_funcs = dict(ssim=MetricFuncGen('ssim'))
    for data in tqdm(val_generator, total=len(val_generator)):
        if ('mask' in data and torch.any(data['mask'])) or args.dataset == 'fastmri':
            scalar_funcs = {
                metric: _FfuncGen(data, model, k_sampler, f)
                for metric, f in metric_funcs.items()
            }
            for metric_name, metric_func in scalar_funcs.items():
                if metric_name.endswith('_vec'):
                    dice_scores = metric_func(pattern)
                    for i, score in enumerate(dice_scores):
                        results[f'{metric_name}_{i}'].append(torch.nan_to_num(score, 0.0).item())
                else:
                    results[metric_name].append(torch.nan_to_num(metric_func(pattern), 0.0).item())
    df = pd.DataFrame.from_dict({c: v for c, v in results.items()})
    df.to_csv(csv_fp, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    log_dir = set_logging(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Load datasets
    train_dataset, val_dataset = get_dataset(
        args.dataset, device,
        preload=args.preload,
        volumetric=True if args.model == 'unet3d' else False
    )
    if args.dataset == 'fastmri_plus':
        train_generator = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=lambda batch: [{k:v for k,v in b.items()} for b in batch],
            batch_size=args.batch, shuffle=True)
        val_generator = torch.utils.data.DataLoader(
            val_dataset,
            collate_fn=lambda batch: [{k:v for k,v in b.items()} for b in batch],
            batch_size=args.val_batch, shuffle=False)
    else:
        train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch, shuffle=False)
    # Get image sizes
    tmp_data = val_dataset[0]
    size_x, size_y = tmp_data['img'].shape[-2:]
    # Set k-space sampler
    f_map = COIL_MAP[args.dataset] if args.multicoil else None
    if args.method.startswith('pilot'):
        if args.pattern:
            k_sampler = PatternGridSampler(f_maps=f_map, shape=[size_x, size_y],
                                           pattern=torch.load(args.pattern), device=device)
        else:
            k_sampler = GridSampler(f_maps=f_map, shape=[size_x, size_y], device=device)
    else:
        k_sampler = PatternSampler(f_maps=f_map, shape=[size_x, size_y], device=device)
    if args.r < 2:
        p_optimizer = train(args, size_x, size_y, device, log_dir, k_sampler, train_generator, val_generator)
        pattern = p_optimizer.get_val_pattern()
        # val(args, pattern, p_optimizer.model, k_sampler, val_generator, csv_fp=os.path.join(log_dir, 'val.csv'))
        if args.dataset == 'brats':
            warnings.warn('In this script validation on BRaTS not implemented correctly, please use train_unet.py')
    elif args.r == 2:  # if train and validate on kfold
        for i, (train_id, val_id) in enumerate(KFold(shuffle=False).split(range(len(train_dataset)))):
            td = torch.utils.data.Subset(train_dataset, train_id)
            vd = torch.utils.data.Subset(train_dataset, val_id)
            tg = torch.utils.data.DataLoader(td, batch_size=args.batch, shuffle=True)
            vg = torch.utils.data.DataLoader(vd, batch_size=1, shuffle=False)
            cv_log_dir = os.path.join(log_dir, f'{i}fold')
            if args.pattern:
                S = get_segmentation_model(
                    args.dataset, args.model, f_path=args.model_path,
                    nchans=args.nchans, nlayers=args.nlayers, device=device
                ).to(device)
                pattern = torch.load(args.pattern).to(device)
            else:
                p_optimizer = train(args, size_x, size_y, device, cv_log_dir, k_sampler, tg, vg)
                pattern = p_optimizer.get_val_pattern()
                S = p_optimizer.model
            val(args, pattern, S, k_sampler, vg, csv_fp=os.path.join(cv_log_dir, 'val.csv'))
            if args.dataset == 'brats':
                warnings.warn('In this script validation on BRaTS not implemented correctly, please use train_unet.py')
    elif args.r == 3:  # if validate on kfold low-resolution pattern
        for i, (train_id, val_id) in enumerate(KFold(shuffle=False).split(range(len(train_dataset)))):
            td = torch.utils.data.Subset(train_dataset, train_id)
            vd = torch.utils.data.Subset(train_dataset, val_id)
            tg = torch.utils.data.DataLoader(td, batch_size=args.batch, shuffle=True)
            vg = torch.utils.data.DataLoader(vd, batch_size=1, shuffle=False)
            cv_log_dir = os.path.join(log_dir, f'{i}fold')
            os.makedirs(cv_log_dir)
            model = get_segmentation_model(args.dataset, args.model, f_path=args.model_path, nchans=args.nchans,
                                           nlayers=args.nlayers, device=device)
            ax = args.acceleration
            pattern = torch.zeros(1, size_x).cuda().float()
            pattern[:, size_x // 2 - (size_x // ax) // 2: size_x // 2 + size_x // ax - (size_x // ax) // 2] = 1
            val(args, pattern, model, k_sampler, vg, csv_fp=os.path.join(cv_log_dir, 'val.csv'))
            torch.save(pattern.cpu(), os.path.join(cv_log_dir, 'pattern.pt'))
            if args.dataset == 'brats':
                warnings.warn('In this script validation on BRaTS not implemented correctly, please use train_unet.py')
    elif args.r == 4:  # if validate on kfold equispaced
        from kspace import EquispacedMaskFunc
        for i, (train_id, val_id) in enumerate(KFold(shuffle=False).split(range(len(train_dataset)))):
            td = torch.utils.data.Subset(train_dataset, train_id)
            vd = torch.utils.data.Subset(train_dataset, val_id)
            tg = torch.utils.data.DataLoader(td, batch_size=args.batch, shuffle=True)
            vg = torch.utils.data.DataLoader(vd, batch_size=1, shuffle=False)
            cv_log_dir = os.path.join(log_dir, f'{i}fold')
            os.makedirs(cv_log_dir)
            model = get_segmentation_model(args.dataset, args.model, f_path=args.model_path, nchans=args.nchans,
                                           nlayers=args.nlayers, device=device)
            ax = int(args.acceleration)
            pattern = torch.tensor(EquispacedMaskFunc([0.02], [ax])((256, 256))[0])[None]
            val(args, pattern, model, k_sampler, vg, csv_fp=os.path.join(cv_log_dir, 'val.csv'))
            torch.save(pattern.cpu(), os.path.join(cv_log_dir, 'pattern.pt'))
            if args.dataset == 'brats':
                warnings.warn('In this script validation on BRaTS not implemented correctly, please use train_unet.py')
    else:
        raise ValueError
