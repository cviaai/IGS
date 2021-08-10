import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseReconstructionModule


class ENet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, decoder_channels=128, dropout=0.1):
        """
        TODO: Enet docs
        :param in_channels:
        :param out_channels:
        :param decoder_channels:
        :param dropout:
        """
        super().__init__()

        self.net = nn.ModuleList([
            DownSampler(in_channels, decoder_channels//8),
            Bottleneck(decoder_channels//8, decoder_channels//2, dropout/10, downsample=True),

            Bottleneck(decoder_channels//2, decoder_channels//2, dropout/10),
            Bottleneck(decoder_channels//2, decoder_channels//2, dropout/10),
            Bottleneck(decoder_channels//2, decoder_channels//2, dropout/10),
            Bottleneck(decoder_channels//2, decoder_channels//2, dropout/10),

            Bottleneck(decoder_channels//2, decoder_channels, dropout, downsample=True),

            Bottleneck(decoder_channels, decoder_channels, dropout),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=2),
            Bottleneck(decoder_channels, decoder_channels, dropout, asymmetric_ksize=5),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=4),
            Bottleneck(decoder_channels, decoder_channels, dropout),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=8),
            Bottleneck(decoder_channels, decoder_channels, dropout, asymmetric_ksize=5),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=16),

            Bottleneck(decoder_channels, decoder_channels, dropout),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=2),
            Bottleneck(decoder_channels, decoder_channels, dropout, asymmetric_ksize=5),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=4),
            Bottleneck(decoder_channels, decoder_channels, dropout),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=8),
            Bottleneck(decoder_channels, decoder_channels, dropout, asymmetric_ksize=5),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=16),

            UpSampler(decoder_channels, decoder_channels//2),

            Bottleneck(decoder_channels//2, decoder_channels//2, dropout),
            Bottleneck(decoder_channels//2, decoder_channels//2, dropout),

            UpSampler(decoder_channels//2, decoder_channels//8),

            Bottleneck(decoder_channels//8, decoder_channels//8, dropout),

            nn.ConvTranspose2d(decoder_channels//8, out_channels, (2, 2), (2, 2))])

    def forward(self, x):
        max_indices_stack = []

        for module in self.net:
            if isinstance(module, UpSampler):
                x = module(x, max_indices_stack.pop())
            else:
                x = module(x)

            if type(x) is tuple:
                x, max_indices = x
                max_indices_stack.append(max_indices)

        return x


class UpSampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        bt_channels = in_channels // 4

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, bt_channels, (1, 1), bias=False),
            nn.BatchNorm2d(bt_channels, 1e-3),
            nn.ReLU(True),

            nn.ConvTranspose2d(bt_channels, bt_channels, (3, 3), 2, 1, 1),
            nn.BatchNorm2d(bt_channels, 1e-3),
            nn.ReLU(True),

            nn.Conv2d(bt_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, 1e-3))

        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, 1e-3))

    def forward(self, x, max_indices):
        x_skip_connection = self.skip_connection(x)
        x_skip_connection = F.max_unpool2d(x_skip_connection, max_indices, (2, 2))

        return F.relu(x_skip_connection + self.main_branch(x), inplace=True)


class DownSampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, (3, 3), 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, 1e-3)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = torch.cat((F.max_pool2d(x, (2, 2)), self.conv(x)), 1)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Bottleneck(nn.Module):
    def __init__(
            self, in_channels, out_channels, dropout_prob=0.0, downsample=False,
            asymmetric_ksize=None, dilation=1, use_prelu=True):

        super().__init__()
        bt_channels = in_channels // 4
        self.downsample = downsample
        self.channels_to_pad = out_channels - in_channels

        input_stride = 2 if downsample else 1

        main_branch = [
            nn.Conv2d(in_channels, bt_channels, input_stride, input_stride, bias=False),
            nn.BatchNorm2d(bt_channels, 1e-3),
            nn.PReLU(bt_channels) if use_prelu else nn.ReLU(True)
        ]

        if asymmetric_ksize is None:
            main_branch += [
                nn.Conv2d(bt_channels, bt_channels, (3, 3), 1, dilation, dilation)
            ]
        else:
            assert type(asymmetric_ksize) is int
            ksize, padding = asymmetric_ksize, (asymmetric_ksize - 1) // 2
            main_branch += [
                nn.Conv2d(bt_channels, bt_channels, (ksize, 1), 1, (padding, 0), bias=False),
                nn.Conv2d(bt_channels, bt_channels, (1, ksize), 1, (0, padding))
            ]

        main_branch += [
            nn.BatchNorm2d(bt_channels, 1e-3),
            nn.PReLU(bt_channels) if use_prelu else nn.ReLU(True),
            nn.Conv2d(bt_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, 1e-3),
            nn.Dropout2d(dropout_prob)
        ]

        self.main_branch = nn.Sequential(*main_branch)
        self.output_activation = nn.PReLU(out_channels) if use_prelu else nn.ReLU(True)

    def forward(self, x):
        if self.downsample:
            x_skip_connection, max_indices = F.max_pool2d(x, (2, 2), return_indices=True)
        else:
            x_skip_connection = x

        if self.channels_to_pad > 0:
            x_skip_connection = F.pad(x_skip_connection, (0, 0, 0, 0, 0, self.channels_to_pad))

        x = self.output_activation(x_skip_connection + self.main_branch(x))

        if self.downsample:
            return x, max_indices
        else:
            return x


class EnetModule(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(EnetModule, self).__init__(**kwargs)

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(1)

    def get_net(self, **kwargs):
        return ENet(
            in_channels=1,
            out_channels=1,
            decoder_channels=kwargs['enet_channels'],
            dropout=kwargs['enet_dropout']
        )