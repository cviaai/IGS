import os
import numpy as np
import h5py

from datetime import datetime
from collections import defaultdict

import torch
import torchvision
import pytorch_lightning as pl
from typing import overload

from k_space_reconstruction.nets.dncnn import DnCNN
from k_space_reconstruction.nets.unet import Unet

from k_space_reconstruction.utils.kspace import pt_kspace2spatial as FtH
from k_space_reconstruction.utils.kspace import pt_spatial2kspace as Ft
from k_space_reconstruction.nets.base import BaseReconstructionModule

from k_space_reconstruction.nets.cddn import DataConsistencyModule, DataConsistencyLLearnableModule, DCSuperAFModuleV2


class CascadeModule(BaseReconstructionModule):

    def __init__(self, net, **kwargs):
        super(CascadeModule, self).__init__(**kwargs)
        self.net = net

    def forward(self, k, m, x, mean, std):
        print(x)
        x = x.unsqueeze(1)
        print(x)
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


class DnCNNDCModule(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(DnCNNDCModule, self).__init__(**kwargs)

    def get_net(self, **kwargs):
        return DnCNNCascade(kwargs['dncnn_chans'], kwargs['dncnn_depth'])


class DnCNNDCLModule(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(DnCNNDCLModule, self).__init__(**kwargs)

    def get_net(self, **kwargs):
        return DnCNNDCLCascade(kwargs['dncnn_chans'], kwargs['dncnn_depth'])


class DnCNNDCsuperAFV2(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(DnCNNDCsuperAFV2, self).__init__(**kwargs)

    def get_net(self, **kwargs):
        return DnCNNDCsuperAFV2Cascade(kwargs['dncnn_chans'], kwargs['dncnn_depth'])


class DnCNNDCLCascade(torch.nn.Module):

    def __init__(self, n_filters, num_layers): # dncnn_chans, dncnn_depth
        super().__init__()
        self.cascade = torch.nn.ModuleList([DnCNN(1, 1, n_filters, num_layers), DataConsistencyLLearnableModule()])

    def forward(self, k, m, x, mean, std):
        for module in self.cascade:
            if type(module) == DataConsistencyLLearnableModule:
                x = module(k, m, x, mean, std)
            else:
                x = module(x)
        return x


class DnCNNDCsuperAFV2Cascade(torch.nn.Module):

    def __init__(self, n_filters, num_layers): # dncnn_chans, dncnn_depth
        super().__init__()
        self.cascade = torch.nn.ModuleList([DnCNN(1, 1, n_filters, num_layers), DCSuperAFModuleV2()])

    def forward(self, k, m, x, mean, std):
        for module in self.cascade:
            if type(module) == DCSuperAFModuleV2:
                x = module(k, m, x, mean, std)
            else:
                x = module(x)
        return x


class DnCNNCascade(torch.nn.Module):

    def __init__(self, n_filters, num_layers): # dncnn_chans, dncnn_depth
        super().__init__()
        self.cascade = torch.nn.ModuleList([DnCNN(1, 1, n_filters, num_layers), DataConsistencyModule()])

    def forward(self, k, m, x, mean, std):
        for module in self.cascade:
            if type(module) == DataConsistencyModule:
                x = module(k, m, x, mean, std)
            else:
                x = module(x)
        return x
    
#-----------DnCNN-without-DC-----------------------------------

class PureDnCNNDCModule(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(PureDnCNNDCModule, self).__init__(**kwargs)

    def get_net(self, **kwargs):
        return PureDnCNNCascade(kwargs['dncnn_chans'], kwargs['dncnn_depth'])

class PureDnCNNCascade(torch.nn.Module):

    def __init__(self, n_filters, num_layers): # dncnn_chans, dncnn_depth
        super().__init__()
        self.cascade = torch.nn.ModuleList([DnCNN(1, 1, n_filters, num_layers)])

    def forward(self, k, m, x, mean, std):
        for module in self.cascade:
            x = module(x)
#             if type(module) == DataConsistencyModule:
#                 x = module(k, m, x, mean, std)
#             else:
#                 x = module(x)
        return x
    
#------------DnCNN+UNet-----------------------------------------
class BothDCModule(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(BothDCModule, self).__init__(**kwargs)

    def get_net(self, **kwargs):
        return BothCascade(kwargs['dncnn_chans'], kwargs['dncnn_depth'])

class BothCascade(torch.nn.Module):

    def __init__(self, n_filters, num_layers): # dncnn_chans, dncnn_depth
        super().__init__()
        self.cascade = torch.nn.ModuleList([DnCNN(1, 1, n_filters, num_layers), DataConsistencyModule()])
        self.cascade_2 = torch.nn.ModuleList([Unet(1, 1, n_filters, num_layers), DataConsistencyLLearnableModule()])
        
    def forward(self, k, m, x, mean, std):
        for module in self.cascade:
            if type(module) == DataConsistencyModule:
                x = module(k, m, x, mean, std)
            else:
                x = module(x)
                
        for module in self.cascade_2:
            if type(module) == DataConsistencyModule:
                x = module(k, m, x, mean, std)
            else:
                x = module(x)
        return x