import os
import re
import h5py
import cv2
import numpy as np
import torch
import torch.utils.data
import pytorch_lightning as pl
from tqdm import tqdm
from typing import Any, Union, List, Optional
from os.path import isdir, join
from torch.utils.data.dataset import T_co
from torch.utils.data import Dataset, DataLoader, random_split
from k_space_reconstruction.utils.kspace import RandomMaskFunc, MaskFunc, spatial2kspace, kspace2spatial, apply_mask
from k_space_reconstruction.utils.io import get_dir_md5hash, get_file_md5hash


class FastMRITransformC(object):

    def __init__(self, mask_f, target_shape=(320, 320), noise_level=0.0, noise_type='none'):
        self.mask_f = mask_f
        self.target_shape = target_shape
        self.noise_level = noise_level
        self.noise_type = noise_type

    @staticmethod
    def normalize(x: np.ndarray):
        mean = x.mean()
        std = x.std()
        x = (x - mean) / (std + 1e-11 + 1j * 1e-11)
        return x, mean, std

    def add_noise(self, ks):
        if self.noise_type == 'none':
            return ks
        elif self.noise_type == 'normal':
            return ks + np.random.normal(size=(ks.shape[0], ks.shape[1])) * ks.mean() * self.noise_level
        elif self.noise_type == 'poisson':
            return ks + np.random.poisson(size=(ks.shape[0], ks.shape[1])) * ks.mean() * self.noise_level
        elif self.noise_type == 'salt':
            shape = ks.shape
            ks = ks.flatten()
            i = np.random.randint(low=0, high=shape[0] * shape[1], size=10)
            ks[i] = ks.mean() * self.noise_level
            ks = ks.reshape(shape)
            return ks

    def __call__(self, f_name: str, slice_id: str, k_space: np.ndarray, max_val: float):
        recon = kspace2spatial(k_space)
        xs = (k_space.shape[0] - self.target_shape[0]) // 2
        ys = (k_space.shape[1] - self.target_shape[1]) // 2
        xt = xs + self.target_shape[0]
        yt = ys + self.target_shape[1]
        recon = recon[xs:xt, ys:yt]
        k_space = spatial2kspace(recon)
        k_space = self.add_noise(k_space)
        if self.mask_f:
            k_space, mask = apply_mask(k_space, self.mask_f)

        r = np.fft.fftshift(k_space)
        r = np.fft.ifftn(r, norm='ortho')
        sampled_image = np.fft.ifftshift(r)

        sampled_image, mean, std = self.normalize(sampled_image)
        target = (recon - mean) / (std + 1e-11)

        k_space = torch.as_tensor(k_space, dtype=torch.complex64).unsqueeze(0)
        mask = torch.as_tensor(mask, dtype=torch.complex64).unsqueeze(0)
        target = torch.as_tensor(target, dtype=torch.complex64).unsqueeze(0)
        sampled_image = torch.as_tensor(sampled_image, dtype=torch.complex64).unsqueeze(0)
        mean = torch.as_tensor(mean, dtype=torch.complex64).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        std = torch.as_tensor(std, dtype=torch.complex64).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        target = (target * std + mean).abs()
        return k_space, mask, target, sampled_image, mean, std, f_name, slice_id, max_val


class FastMRITransform(object):

    def __init__(self, mask_f: MaskFunc, target_shape=(320, 320), noise_level=0.0, noise_type='none'):
        self.mask_f = mask_f
        self.target_shape = target_shape
        self.noise_level = noise_level
        self.noise_type = noise_type

    @staticmethod
    def normalize(x: np.ndarray):
        mean = x.mean()
        std = x.std()
        x = (x - mean) / (std + 1e-11)
        return x, mean, std

    def add_noise(self, ks):
        if self.noise_type == 'none':
            return ks
        elif self.noise_type == 'normal':
            return ks + np.random.normal(size=(ks.shape[0], ks.shape[1])) * ks.mean() * self.noise_level
        elif self.noise_type == 'poisson':
            return ks + np.random.poisson(size=(ks.shape[0], ks.shape[1])) * ks.mean() * self.noise_level
        elif self.noise_type == 'salt':
            shape = ks.shape
            ks = ks.flatten()
            i = np.random.randint(low=0, high=shape[0] * shape[1], size=10)
            ks[i] = ks.mean() * self.noise_level
            ks = ks.reshape(shape)
            return ks

    def __call__(self, f_name: str, slice_id: str, k_space: np.ndarray, max_val: float):
        recon = kspace2spatial(k_space)
        xs = (k_space.shape[0] - self.target_shape[0]) // 2
        ys = (k_space.shape[1] - self.target_shape[1]) // 2
        xt = xs + self.target_shape[0]
        yt = ys + self.target_shape[1]
        recon = recon[xs:xt, ys:yt]
        k_space = spatial2kspace(recon)
        k_space = self.add_noise(k_space)
        if self.mask_f:
            k_space, mask = apply_mask(k_space, self.mask_f)
        sampled_image = kspace2spatial(k_space)
        sampled_image, mean, std = self.normalize(sampled_image)
        target = (recon - mean) / (std + 1e-11)

        k_space = torch.as_tensor(np.stack((k_space.real, k_space.imag)), dtype=torch.float)
        mask = torch.as_tensor(mask, dtype=torch.float).unsqueeze(0)
        target = torch.as_tensor(target, dtype=torch.float).unsqueeze(0)
        sampled_image = torch.as_tensor(sampled_image, dtype=torch.float).unsqueeze(0)
        mean = torch.as_tensor(mean, dtype=torch.float).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        std = torch.as_tensor(std, dtype=torch.float).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return k_space, mask, target, sampled_image, mean, std, f_name, slice_id, max_val


class FastMRIDataset(Dataset):

    def __init__(self, dir_path, transform):
        super(FastMRIDataset, self).__init__()
        self.dir = dir_path
        self.transform = transform
        self._slices = []
        for f in sorted(os.listdir(self.dir)):
            hf = h5py.File(join(self.dir, f))
            for iz in range(hf['kspace'].shape[0]):
                self._slices += [(join(self.dir, f), iz)]
            hf.close()

    def __len__(self):
        return len(self._slices)

    def __getitem__(self, index) -> T_co:
        fp, slice_id = self._slices[index]
        hf = h5py.File(fp)
        ks = hf['kspace'][:]
        # ks = ks * 1e6
        maxval = np.stack([kspace2spatial(k) for k in ks]).max()
        # maxval = hf.attrs['max']
        # target = hf['reconstruction_esc'][slice_id]
        if self.transform:
            return self.transform(fp, slice_id, ks[slice_id], maxval)
        else:
            return torch.as_tensor(np.stack((ks.real, ks.imag)), dtype=torch.float)


class FastMRIh5Dataset(Dataset):

    def __init__(self, hf_path, transform):
        super(FastMRIh5Dataset, self).__init__()
        self.hf_path = hf_path
        self.hf = h5py.File(hf_path)
        self.transform = transform
        self._slices = []
        for f in sorted(list(self.hf.keys())):
            for iz in range(self.hf[f].shape[0]):
                self._slices += [(f, iz)]

    def __len__(self):
        return len(self._slices)

    def __getitem__(self, index) -> T_co:
        key, slice_id = self._slices[index]
        ks = self.hf[key][:]
        ks = ks * 1e6
        maxval = np.stack([kspace2spatial(k) for k in ks]).max()
        if self.transform:
            return self.transform(key, slice_id, ks[slice_id], maxval)
        else:
            return torch.as_tensor(np.stack((ks.real, ks.imag)), dtype=torch.float)


class LegacyFastMRIh5Dataset(Dataset):

    def __init__(self, hf_path, transform):
        super(LegacyFastMRIh5Dataset, self).__init__()
        self.hf_path = hf_path
        self.hf = h5py.File(hf_path)
        self.transform = transform
        self._slices = []
        for f in sorted(list(self.hf.keys())):
            for iz in range(self.hf[f].shape[0]):
                self._slices += [(f, iz)]

    def __len__(self):
        return len(self._slices)

    def __getitem__(self, index) -> T_co:
        key, slice_id = self._slices[index]
        if self.transform:
            ks = self.hf[key][:]
            maxval = np.stack([kspace2spatial(k) for k in ks]).max()
            return self.transform(key, slice_id, ks[slice_id], maxval)
        else:
            ks = self.hf[key][slice_id]
            xs = (ks.shape[0] - 640) // 2
            ys = (ks.shape[1] - 320) // 2
            xt = xs + 640
            yt = ys + 320
            return torch.as_tensor(self.hf[key][slice_id][xs:xt, ys:yt])


class PlFastMRIkneeDataModule(pl.LightningDataModule):
    DIR_NAME = 'fastMRIdatasets'
    DIR_TRAIN = 'singlecoil_train'
    DIR_VAL = 'singlecoil_val'
    DIR_TRAIN_HASH = '00651c3286630fe70d8aaaa119564c67'
    DIR_VAL_HASH = '9562281616da52c6aac67bb6b9132053'

    def __init__(self, root_dir, transform, batch_size=1, num_workers=0, prefetch_factor=2, random_seed=42, train_val_split=0.2):
        super(PlFastMRIkneeDataModule, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self._train = None
        self._val = None
        self._test = None
        self.random_seed = random_seed

    # def prepare_data(self, *args, **kwargs):
    #     if os.path.exists(join(self.root_dir, self.DIR_NAME)) and os.path.isdir(join(self.root_dir, self.DIR_NAME)):
    #         if not get_dir_md5hash(join(self.root_dir, self.DIR_NAME, self.DIR_TRAIN)) == self.DIR_TRAIN_HASH:
    #             raise ValueError('Wrong checksum, delete %s dir' % self.root_dir)
    #         if not get_dir_md5hash(join(self.root_dir, self.DIR_NAME, self.DIR_TEST)) == self.DIR_TEST_HASH:
    #             raise ValueError('Wrong checksum, delete %s dir' % self.root_dir)
    #         return True
    #     raise ValueError('Dir not exist')

    def setup(self, stage: Optional[str] = None):
        self._train = FastMRIDataset(join(self.root_dir, self.DIR_NAME, self.DIR_TRAIN), self.transform)
        self._val = FastMRIDataset(join(self.root_dir, self.DIR_NAME, self.DIR_VAL), self.transform)
        # TODO: ?
        self._test = FastMRIDataset(join(self.root_dir, self.DIR_NAME, self.DIR_VAL), self.transform)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self._train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self._val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self._test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)


class PlFastMRIkneeH5DataModule(PlFastMRIkneeDataModule):
    DIR_NAME = 'fastMRIh5'
    HF_TRAIN = 'singlecoil_train.h5'
    HF_VAL = 'singlecoil_val.h5'

    def setup(self, stage: Optional[str] = None):
        self._train = FastMRIh5Dataset(join(self.root_dir, self.DIR_NAME, self.HF_TRAIN), self.transform)
        self._val = FastMRIh5Dataset(join(self.root_dir, self.DIR_NAME, self.HF_VAL), self.transform)
        self._test = FastMRIh5Dataset(join(self.root_dir, self.DIR_NAME, self.HF_VAL), self.transform)


if __name__ == '__main__':
    dataset = FastMRIDataset('/home/a_razumov/smbmount_a_razumov/fastMRIdatasets/singlecoil_val',
                               FastMRITransform(RandomMaskFunc([0.08], [1])))
    print(len(dataset))
    ks, mask, y, x, mean, std, _, _, _ = dataset[20]
    print(ks.shape)

    import pylab as plt
    fig, ax = plt.subplots(nrows=2, figsize=(5, 4 * 2),
                           subplot_kw=dict(frameon=False, xticks=[], yticks=[]),
                           gridspec_kw=dict(wspace=0.0, hspace=0.0))
    ax[0].imshow(y[0])
    ax[1].imshow(((ks[0] + 1j*ks[1]).abs() + 1e-11).log())
    print(np.linalg.norm(y[0] - x[0]))
    plt.show()
    print(ks.shape, x.shape)
    print(ks.mean(), ks.std(), ks.max(), ks.min())
    print(x.mean(), x.std(), x.max(), x.min())
    print(np.linalg.norm(y.numpy() - x.numpy()))

