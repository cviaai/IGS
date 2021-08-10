import torch
import torch.nn.functional as F
from k_space_reconstruction.nets.base import BaseReconstructionModule
from k_space_reconstruction.nets.unet import Unet
from k_space_reconstruction.utils.kspace import pt_spatial2kspace, pt_kspace2spatial


class ActiveLayer(torch.nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.w_1 = torch.nn.Parameter(data=torch.ones(shape))
        self.b_1 = torch.nn.Parameter(data=torch.ones(shape) * 1e-1)
        self.w_2 = torch.nn.Parameter(data=torch.ones(shape))
        self.b_2 = torch.nn.Parameter(data=torch.ones(shape) * 1e-1)

    def forward(self, img, mean, std):
        ks = pt_spatial2kspace(img * std + mean)
        x = F.relu(F.relu(self.w_1) * ks.abs() + F.relu(self.b_1))
        x = F.relu(self.w_2) * x + F.relu(self.b_2)
        ks = ks * x.abs() / (ks.abs() + 1e-11)
        y = pt_kspace2spatial(ks).abs()
        y = (y - mean) / (std + 1e-11)
        return y


class UnetAFModule(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(UnetAFModule, self).__init__(**kwargs)

    def forward(self, x, mean, std):
        x = self.net[0](x.unsqueeze(1), mean, std)
        return self.net[1](x).squeeze(1)

    def predict(self, batch):
        ks, mask, y, x, mean, std, f_name, slice_id, max_val = batch
        x = self.net[0](x, mean, std)
        return self.net[1](x)

    def get_net(self, **kwargs):
        return torch.nn.ModuleList([
            ActiveLayer(shape=kwargs['al_shape']),
            Unet(in_chans=1, out_chans=1, num_pool_layers=kwargs['unet_num_layers'], chans=kwargs['unet_chans'])
        ])
