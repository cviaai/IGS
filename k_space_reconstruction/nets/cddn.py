import torch
import torch.nn as nn
import torch.nn.functional as F
from k_space_reconstruction.nets.base import BaseReconstructionModule
from k_space_reconstruction.utils.kspace import pt_kspace2spatial as FtH
from k_space_reconstruction.utils.kspace import pt_spatial2kspace as Ft


PADDING_MODE = 'zeros'


class CDDNwTDCModule(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(CDDNwTDCModule, self).__init__(**kwargs)

    def forward(self, k, m, x, mean, std):
        return self.net(k, m.unsqueeze(1), x.unsqueeze(1), mean.unsqueeze(1), std.unsqueeze(1)).squeeze(1)

    def get_net(self, **kwargs):
        return CDDNwTDC(
            in_channels=1,
            n_filters=kwargs['cddn_n_filters'],
            n_cascades=kwargs['cddn_n_cascades']
        )

    def predict(self, batch):
        ks, mask, y, x, mean, std, f_name, slice_id, max_val = batch
        return self.net(ks, mask, x, mean, std)


class CDDNwTDC(nn.Module):

    def __init__(self, in_channels, n_filters, n_cascades):
        super().__init__()
        self.n_cascades = n_cascades
        self.cascades = nn.ModuleList(
            [DAMModule(in_channels, n_filters), TDCModule()] * n_cascades
        )

    def forward(self, k, m, x, mean, std):
        for module in self.cascades:
            if type(module) == DAMModule:
                x = module(x)
            elif type(module) == TDCModule:
                x = module(k, m, x, mean, std)
        return x


class DAMModule(nn.Module):

    def __init__(self, in_channels, n_filters):
        super(DAMModule, self).__init__()
        self.abstruction_layer = AbstractionLayer(in_channels, n_filters)
        self.dense_block = DenseDilatedBlock(n_filters)
        self.transition_layer = TransitionBlock(n_filters * 8)
        self.restore_layer = RestoreBlock(n_filters * 4, in_channels)

    def forward(self, x):
        x = self.abstruction_layer(x)
        x = self.dense_block(x)
        x = self.transition_layer(x)
        x = self.restore_layer(x)
        return x


class TDCModule(nn.Module):

    def __init__(self):
        super(TDCModule, self).__init__()
        self.dc1 = DataConsistencyModule()
        # self.dc2 = DataConsistencyModule()

    def forward(self, k, m, x, mean, std):
        x = self.dc1(k, m, x, mean, std)
        # x = x.abs()
        # x = self.dc2(k, m, x, mean, std)
        return x


class DataConsistencyModule(nn.Module):

    def __init__(self):
        super(DataConsistencyModule, self).__init__()

    def forward(self, k, m, x, mean, std):
        ks = Ft(x*std + mean)
        k = k[:, :1] + 1j * k[:, 1:]
        x = FtH((1 - m) * ks + m * k).abs()
        return (x - mean) / (std + 1e-11)


class DataConsistencyLLearnableModule(nn.Module):

    def __init__(self):
        super(DataConsistencyLLearnableModule, self).__init__()
        self.ll = nn.Parameter(data=torch.tensor(1.0), requires_grad=True)

    def forward(self, k, m, x, mean, std):
        ks = Ft(x*std + mean)
        k = k[:, :1] + 1j * k[:, 1:]
        x = FtH((1 - m) * ks + m * (ks + self.ll * k) / (1 + self.ll)).abs()
        return (x - mean) / (std + 1e-11)


class AbstractionLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AbstractionLayer, self).__init__()
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(3, 3),
            padding=1,
            padding_mode=PADDING_MODE,
            bias=False
        )

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))
        # x = self.norm(x)
        # x = self.activation(x)
        # return self.conv(x)


class DenseConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation=1):
        super(DenseConvBlock, self).__init__()
        self.norm1x1 = nn.InstanceNorm2d(in_channels//2)
        self.activation1x1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1x1 = nn.Conv2d(
            in_channels, in_channels//2,
            kernel_size=(1, 1),
            bias=False
        )
        self.norm3x3 = nn.InstanceNorm2d(out_channels)
        self.activation3x3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv3x3 = nn.Conv2d(
            in_channels//2, out_channels,
            kernel_size=(3, 3),
            padding=dilation,
            dilation=dilation,
            padding_mode=PADDING_MODE,
            bias=False
        )

    def forward(self, x):
        x = self.activation1x1(self.norm1x1(self.conv1x1(x)))
        x = self.activation3x3(self.norm3x3(self.conv3x3(x)))
        # x = self.conv1x1(self.activation1x1(self.norm1x1(x)))
        # x = self.conv3x3(self.activation3x3(self.norm3x3(x)))
        return x


class DenseDilatedBlock(nn.Module):

    def __init__(self, in_channels):
        super(DenseDilatedBlock, self).__init__()
        self.block1 = DenseConvBlock(in_channels, in_channels, dilation=1)
        self.block2 = DenseConvBlock(in_channels*2, in_channels*2, dilation=2)
        self.block3 = DenseConvBlock(in_channels*4, in_channels*4, dilation=4)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(torch.cat([x, x1], dim=1))
        x3 = self.block3(torch.cat([x, x1, x2], dim=1))
        return torch.cat([x, x1, x2, x3], dim=1)


class TransitionBlock(nn.Module):

    def __init__(self, in_channels):
        super(TransitionBlock, self).__init__()
        self.norm = nn.InstanceNorm2d(in_channels//2)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv = nn.Conv2d(
            in_channels, in_channels//2,
            kernel_size=(1, 1),
            bias=False
        )

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))
        # return self.conv(self.activation(self.norm(x)))


class RestoreBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(RestoreBlock, self).__init__()
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(3, 3),
            padding=1,
            padding_mode=PADDING_MODE,
            bias=False
        )

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))
        # return self.conv(self.activation(self.norm(x)))


if __name__ == '__main__':
    x = torch.rand(2, 1, 320, 320)
    m = torch.rand(2, 1, 1, 320)
    k = torch.rand(2, 2, 320, 320)
    mean = torch.rand(2, 1, 1, 1)
    std = torch.rand(2, 1, 1, 1)

    net = CDDNwTDC(1, 16, 5)
    print(net(k, m, x, mean, std).shape)

    from torchsummary import summary
    summary(net.cuda(), input_size=[(2, 256, 256), (1, 1, 256), (1, 256, 256), (1, 1, 1), (1, 1, 1)])
    # net = DAMModule(1, 16)
    # summary(net.cuda(), input_size=(1, 256, 256))