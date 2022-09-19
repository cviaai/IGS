import os
import numpy as np
import h5py

from datetime import datetime
from collections import defaultdict

import torch
import torchvision
import pytorch_lightning as pl
from typing import overload
from k_space_reconstruction.utils.kspace import pt_kspace2spatial as FtH
from k_space_reconstruction.utils.kspace import pt_spatial2kspace as Ft
from k_space_reconstruction.nets.base import BaseReconstructionModule
from k_space_reconstruction.nets.kunet import KUnet
from k_space_reconstruction.nets.mwcnn import MWCNN
from k_space_reconstruction.nets.unet import Unet
from k_space_reconstruction.nets.cddn import DataConsistencyModule, DataConsistencyLLearnableModule
from k_space_reconstruction.nets.cddn import DLAFModule, DCSuperAFModule, DCSuperAFModuleV2, DCSuperAFModuleV3, DCSuperAFModuleV4


class ComplexModule(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(ComplexModule, self).__init__(**kwargs)

    def forward(self, k, m, x, mean, std):
        x = x.unsqueeze(1)
        x = self.net(k, m.unsqueeze(1), x, mean.unsqueeze(1), std.unsqueeze(1))
        return (x*std + mean).abs().squeeze(1)

    def predict(self, batch):
        ks, mask, y, x, mean, std, f_name, slice_id, max_val = batch
        x = self.net(ks, mask, x, mean, std)
        return (x*std + mean).abs()

    def validation_step(self, batch, batch_idx):
        ks, mask, y, x, mean, std, f_name, slice_id, max_val = batch
        yp = self.predict(batch)
        loss = self.criterion(yp, y)
        return {
            'batch_idx': batch_idx,
            'f_name': f_name,
            'slice_id': slice_id,
            'max_val': max_val,
            'input': (x * std + mean).abs(),
            'output': yp,
            'target': y,
            'val_loss': loss
        }

    def test_step(self, batch, batch_idx):
        ks, mask, y, x, mean, std, f_name, slice_id, max_val = batch
        yp = self.predict(batch)
        return {
            'f_name': f_name,
            'slice_id': slice_id,
            'output': yp.cpu().numpy()
        }


class C_CascadeModule(ComplexModule):

    def __init__(self, net, **kwargs):
        super(C_CascadeModule, self).__init__(**kwargs)
        self.net = net

    def forward(self, k, m, x, mean, std):
        x = x.unsqueeze(1)
        for cascade in self.net:
            x = cascade(k, m.unsqueeze(1), x, mean.unsqueeze(1), std.unsqueeze(1))
        return (x*std + mean).abs().squeeze(1)

    def predict(self, batch):
        ks, mask, y, x, mean, std, f_name, slice_id, max_val = batch
        for cascade in self.net:
            x = cascade(ks, mask, x, mean, std)
        return (x*std + mean).abs()

    def get_net(self, **kwargs):
        return None

class CascadeModule(BaseReconstructionModule):

    def __init__(self, net, **kwargs):
        super(CascadeModule, self).__init__(**kwargs)
        self.net = net

    def forward(self, k, m, x, mean, std):
        x = x.unsqueeze(1)
        for cascade in self.net:
            x = cascade(k, m.unsqueeze(1), x, mean.unsqueeze(1), std.unsqueeze(1))
        return x

    def predict(self, batch):
        ks, mask, y, x, mean, std, f_name, slice_id, max_val = batch
        for cascade in self.net:
            x = cascade(ks, mask, x, mean, std)
        return x

    def get_net(self, **kwargs):
        return None


class C_KUnetDCModule(ComplexModule):

    def __init__(self, **kwargs):
        super(C_KUnetDCModule, self).__init__(**kwargs)

    def get_net(self, **kwargs):
        return C_KUnetCascade(kwargs['unet_chans'], kwargs['unet_num_layers'])


class UnetDCModule(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(UnetDCModule, self).__init__(**kwargs)

    def get_net(self, **kwargs):
        return UnetDCCascade(kwargs['unet_chans'], kwargs['unet_num_layers'])

class UnetDCLModule(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(UnetDCLModule, self).__init__(**kwargs)

    def get_net(self, **kwargs):
        return UnetDCLCascade(kwargs['unet_chans'], kwargs['unet_num_layers'])

class UnetDCsuperAFModule(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(UnetDCsuperAFModule, self).__init__(**kwargs)

    def get_net(self, **kwargs):
        return UnetDCsuperAFCascade(kwargs['unet_chans'], kwargs['unet_num_layers'])

class UnetDCsuperAFV2Module(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(UnetDCsuperAFV2Module, self).__init__(**kwargs)

    def get_net(self, **kwargs):
        return UnetDCsuperAFV2Cascade(kwargs['unet_chans'], kwargs['unet_num_layers'])

class UnetDCsuperAFV3Module(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(UnetDCsuperAFV3Module, self).__init__(**kwargs)

    def get_net(self, **kwargs):
        return UnetDCsuperAFV3Cascade(kwargs['unet_chans'], kwargs['unet_num_layers'])

class UnetDCsuperAFV4Module(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(UnetDCsuperAFV4Module, self).__init__(**kwargs)

    def get_net(self, **kwargs):
        return UnetDCsuperAFV4Cascade(kwargs['unet_chans'], kwargs['unet_num_layers'])

class UnetDCAFModule(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(UnetDCAFModule, self).__init__(**kwargs)

    def get_net(self, **kwargs):
        return UnetDCAFCascade(kwargs['unet_chans'], kwargs['unet_num_layers'])

class MWCCNDCModule(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(MWCCNDCModule, self).__init__(**kwargs)

    def get_net(self, **kwargs):
        return MWCCNCascade(kwargs['mwcnn_n_feats'])


class C_DCModule(torch.nn.Module):

    def __init__(self):
        super(C_DCModule, self).__init__()

    def forward(self, k, m, x, mean, std):
        ks = Ft(x * std + mean)
        x = FtH((1 - m) * ks + m * k)
        return (x - mean) / (std + 1e-11 + 1j * 1e-11)


class C_KUnetCascade(torch.nn.Module):

    def __init__(self, n_filters, num_layers):
        super().__init__()
        self.cascade = torch.nn.ModuleList([KUnet(1, 1, n_filters, num_layers), C_DCModule()])

    def forward(self, k, m, x, mean, std):
        for module in self.cascade:
            if type(module) == C_DCModule:
                x = module(k, m, x, mean, std)
            else:
                x = module(x)
        return x


class UnetDCCascade(torch.nn.Module):

    def __init__(self, n_filters, num_layers):
        super().__init__()
        self.cascade = torch.nn.ModuleList([Unet(1, 1, n_filters, num_layers), DataConsistencyModule()])

    def forward(self, k, m, x, mean, std):
        for module in self.cascade:
            if (type(module) == DataConsistencyModule):
                x = module(k, m, x, mean, std)
            else:
                x = module(x)
        return x


class UnetDCsuperAFCascade(torch.nn.Module):

    def __init__(self, n_filters, num_layers):
        super().__init__()
        self.cascade = torch.nn.ModuleList([Unet(1, 1, n_filters, num_layers), DCSuperAFModule()])

    def forward(self, k, m, x, mean, std):
        for module in self.cascade:
            if (type(module) == DCSuperAFModule):
                x = module(k, m, x, mean, std)
            else:
                x = module(x)
        return x


class UnetDCsuperAFV2Cascade(torch.nn.Module):

    def __init__(self, n_filters, num_layers):
        super().__init__()
        self.cascade = torch.nn.ModuleList([Unet(1, 1, n_filters, num_layers), DCSuperAFModuleV2()])

    def forward(self, k, m, x, mean, std):
        for module in self.cascade:
            if (type(module) == DCSuperAFModuleV2):
                x = module(k, m, x, mean, std)
            else:
                x = module(x)
        return x


class UnetDCsuperAFV3Cascade(torch.nn.Module):

    def __init__(self, n_filters, num_layers):
        super().__init__()
        self.cascade = torch.nn.ModuleList([Unet(1, 1, n_filters, num_layers), DCSuperAFModuleV3()])

    def forward(self, k, m, x, mean, std):
        for module in self.cascade:
            if (type(module) == DCSuperAFModuleV3):
                x = module(k, m, x, mean, std)
            else:
                x = module(x)
        return x


class UnetDCsuperAFV4Cascade(torch.nn.Module):

    def __init__(self, n_filters, num_layers):
        super().__init__()
        self.cascade = torch.nn.ModuleList([Unet(1, 1, n_filters, num_layers), DCSuperAFModuleV4()])

    def forward(self, k, m, x, mean, std):
        for module in self.cascade:
            if (type(module) == DCSuperAFModuleV4):
                x = module(k, m, x, mean, std)
            else:
                x = module(x)
        return x


class UnetDCAFCascade(torch.nn.Module):

    def __init__(self, n_filters, num_layers):
        super().__init__()
        self.cascade = torch.nn.ModuleList([Unet(1, 1, n_filters, num_layers), DLAFModule()])

    def forward(self, k, m, x, mean, std):
        for module in self.cascade:
            if (type(module) == DLAFModule):
                x = module(k, m, x, mean, std)
            else:
                x = module(x)
        return x


class UnetDCLCascade(torch.nn.Module):

    def __init__(self, n_filters, num_layers):
        super().__init__()
        self.cascade = torch.nn.ModuleList([Unet(1, 1, n_filters, num_layers), DataConsistencyLLearnableModule()])

    def forward(self, k, m, x, mean, std):
        for module in self.cascade:
            if (type(module) == DataConsistencyLLearnableModule):
                x = module(k, m, x, mean, std)
            else:
                x = module(x)
        return x


class MWCCNCascade(torch.nn.Module):

    def __init__(self, n_filters):
        super().__init__()
        self.cascade = torch.nn.ModuleList([MWCNN(n_filters, 1), DataConsistencyLLearnableModule()])

    def forward(self, k, m, x, mean, std):
        for module in self.cascade:
            if type(module) == DataConsistencyLLearnableModule:
                x = module(k, m, x, mean, std)
            else:
                x = module(x)
        return x
