import torch
import numpy as np


class LinearFourier2d(torch.nn.Module):
    def __init__(self, image_size, log=False):
        super(LinearFourier2d, self).__init__()

        self.log = log

        c, h, w = image_size
        self.register_parameter(name='fourier_filter', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))
        torch.nn.init.ones_(self.fourier_filter)

    def forward(self, x):
        w = torch.nn.ReLU()(self.fourier_filter.repeat(x.shape[0], 1, 1, 1).to(x.device))

        rft_x = torch.rfft(x, signal_ndim=3, normalized=True, onesided=True)
        init_spectrum = torch.sqrt(torch.pow(rft_x[..., 0], 2) + torch.pow(rft_x[..., 1], 2))

        if self.log:
            spectrum = torch.exp(w * torch.log(1 + init_spectrum)) - 1
        else:
            spectrum = w * init_spectrum

        irf = torch.irfft(torch.stack([rft_x[..., 0] * spectrum / (init_spectrum + 1e-16),
                                       rft_x[..., 1] * spectrum / (init_spectrum + 1e-16)], dim=-1),
                          signal_ndim=3, normalized=True, onesided=True, signal_sizes=x.shape[1:])

        return irf


class GeneralFourier2d(torch.nn.Module):
    def __init__(self, image_size, log=False):
        super(GeneralFourier2d, self).__init__()

        self.log = log

        c, h, w = image_size
        self.register_parameter(name='W1', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))

        self.register_parameter(name='B1', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))
        self.register_parameter(name='W2', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))
        self.register_parameter(name='B2', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))

        torch.nn.init.ones_(self.W1)
        torch.nn.init.zeros_(self.B1)
        torch.nn.init.ones_(self.W2)
        torch.nn.init.zeros_(self.B2)

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        w1 = torch.nn.ReLU()(self.W1.repeat(x.shape[0], 1, 1, 1).to(x.device))
        w2 = torch.nn.ReLU()(self.W2.repeat(x.shape[0], 1, 1, 1).to(x.device))
        b1 = torch.nn.ReLU()(self.B1.repeat(x.shape[0], 1, 1, 1).to(x.device))
        b2 = torch.nn.ReLU()(self.B2.repeat(x.shape[0], 1, 1, 1).to(x.device))

        rft_x = torch.rfft(x, signal_ndim=3, normalized=True, onesided=True)
        init_spectrum = torch.sqrt(torch.pow(rft_x[..., 0], 2) + torch.pow(rft_x[..., 1], 2))

        if self.log:
            spectrum = w2 * self.activation(w1 * torch.log(1 + init_spectrum) + b1) + b2
        else:
            spectrum = w2 * self.activation(w1 * init_spectrum + b1) + b2

        irf = torch.irfft(torch.stack([rft_x[..., 0] * spectrum / (init_spectrum + 1e-16),
                                       rft_x[..., 1] * spectrum / (init_spectrum + 1e-16)], dim=-1),
                          signal_ndim=3, normalized=True, onesided=True, signal_sizes=x.shape[1:])

        return irf