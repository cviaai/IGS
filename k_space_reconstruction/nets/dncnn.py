# 1. 
# 2. 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from k_space_reconstruction.nets.base import BaseReconstructionModule


"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

#------------------DnCNN------------------------------------------
class DnCNN(nn.Module):
    """
    PyTorch implementation of Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising 
    https://arxiv.org/pdf/1608.03981.pdf
    https://github.com/cszn/DnCNN/blob/8b61f7e23a68180f5f27002539d745256bd86df2/TrainingCodes/dncnn_pytorch/main_test.py
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        dncnn_chans: int = 64,
        dncnn_depth: int = 10,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the model.
            out_chans: Number of channels in the output to the model.
            dncnn_chans: Number of output channels of the hidden layers.
            dncnn_depth: The depth of DnCNN model
        """
        super().__init__()

        self.in_chans = in_chans # 1
        self.out_chans = out_chans # 1
        self.dncnn_chans = dncnn_chans
        self.dncnn_depth = dncnn_depth

        layers = []
        layers.append(nn.Conv2d(in_channels=in_chans, out_channels=dncnn_chans, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(dncnn_depth - 2):
            layers.append(HidConvBlock(n_chans = dncnn_chans))
            
        layers.append(nn.Conv2d(in_channels=dncnn_chans, out_channels=out_chans, kernel_size=3, padding=1, bias=False))
        self.layers = nn.Sequential(*layers)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        # Residual Learning
        y = image
        out = self.layers(image)
        return y - out

#------------------Hidden-Convolutional-Block----------------------------------------
class HidConvBlock(nn.Module):
    """
    A Convolutional Block that consists of convolution layers with equal number
    of channels, followed by batch normalization and ReLU activation.
    """

    def __init__(self, n_chans: int):
        """
        Args:
            n_chans: Number of channels in the input and output.
        """
        super().__init__()

        self.n_chans = n_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = n_chans, out_channels = self.n_chans, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(n_chans, eps=0.0001, momentum=0.95),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class DnCNNModule(BaseReconstructionModule):

    def __init__(self, **kwargs):
        super(DnCNNModule, self).__init__(**kwargs)

    def forward(self, x):
        return self.net(x)

    def get_net(self, **kwargs): # What to init. Ask Arttem
        return DnCNN(
            in_chans=1,
            out_chans=1,
            dncnn_chans=kwargs['dncnn_chans'], # Number of hidden channels
            dncnn_depth = kwargs['dncnn_depth'],
        )