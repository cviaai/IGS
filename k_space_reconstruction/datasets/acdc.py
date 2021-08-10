import os
import re
import cv2
import gdown
import numpy as np
import nibabel
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


class ACDCTransform(object):

    def __init__(self, mask_f: MaskFunc, target_shape=(320, 320)):
        self.mask_f = mask_f
        self.target_shape = target_shape

    @staticmethod
    def normalize(x: np.ndarray):
        mean = x.mean()
        std = x.std()
        x = (x - mean) / (std + 1e-11)
        return x, mean, std

    def __call__(self, f_name: str, slice_id: str, k_space: np.ndarray, max_val: float):
        recon = kspace2spatial(k_space)
        xs = (k_space.shape[0] - self.target_shape[0]) // 2
        ys = (k_space.shape[1] - self.target_shape[1]) // 2
        xt = xs + self.target_shape[0]
        yt = ys + self.target_shape[1]
        recon = recon[xs:xt, ys:yt]
        k_space = spatial2kspace(recon)
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


class ACDCSet(Dataset):
    IMG4D_PATTERN = r'^patient\d+_4d.nii.gz'

    def __init__(self, dir_path, transform: ACDCTransform):
        super().__init__()
        self.dir = dir_path
        self.transform = transform
        self._images = []
        patients = [d for d in os.listdir(dir_path) if not d.startswith('.') and isdir(join(dir_path, d))]
        for patient in sorted(patients):
            for f in sorted(os.listdir(join(self.dir, patient))):
                if re.findall(self.IMG4D_PATTERN, f):
                    self._images += self.load_scan_slice(join(self.dir, patient, f))

    @staticmethod
    def load_scan_slice(f: str):
        scan = nibabel.load(f)
        images = []
        for i in range(scan.shape[2]):
            for j in range(scan.shape[3]):
                images.append((f, i, j))
        return images

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index) -> T_co:
        f, slice_z, t = self._images[index]
        img = nibabel.load(f).dataobj[:, :, :, t]
        img = img.astype(np.float) / img.max()
        img = cv2.resize(img, (400, 400))
        ks = np.stack([spatial2kspace(img[:, :, i]) for i in range(img.shape[2])])
        ks = ks * 1e2
        maxval = np.stack([kspace2spatial(k) for k in ks]).max()
        if self.transform:
            return self.transform(f, '%d_%d' % (slice_z, t), ks[slice_z], maxval)
        else:
            return torch.as_tensor(img, dtype=torch.float)


class PlACDCDataModule(pl.LightningDataModule):
    DIR_NAME = 'acdc'
    DIR_TRAIN = 'training'
    DIR_TEST = join('testing', 'testing')
    DIR_HASH = 'cb0290e1efb2f752edfd48165db5d04c'
    TAR_HASH = '397d550418e639b722faa95a671986b9'

    def __init__(self, root_dir, transform, batch_size=1, num_workers=0, prefetch_factor=2, random_seed=42, train_val_split=0.2):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self._train = None
        self._val = None
        self._test = None
        self.random_seed = random_seed
        self.train_val_split = train_val_split

    def prepare_data(self):
        if os.path.exists(join(self.root_dir, self.DIR_NAME)) and os.path.isdir(join(self.root_dir, self.DIR_NAME)):
            if get_dir_md5hash(join(self.root_dir, self.DIR_NAME)) == self.DIR_HASH:
                return True
            raise ValueError('Wrong checksum, delete %s dir' % self.root_dir)
        gdown.cached_download('https://drive.google.com/uc?id=1-yZki-hyVcHKWB4VfqAUaAfBazPyemqN',
                              os.path.join(self.root_dir, 'acdc.tar.gz'),
                              md5=self.TAR_HASH, postprocess=gdown.extractall)
        return True

    def setup(self, stage: Optional[str] = None):
        self._train = ACDCSet(join(self.root_dir, self.DIR_NAME, self.DIR_TRAIN), self.transform)
        self._val = ACDCSet(join(self.root_dir, self.DIR_NAME, self.DIR_TEST), self.transform)
        # TODO: ?
        self._test = ACDCSet(join(self.root_dir, self.DIR_NAME, self.DIR_TEST), self.transform)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self._train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self._val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self._test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)


if __name__ == '__main__':
    dataset = ACDCSet('/mnt/dm-3/datasets/acdc/training', ACDCTransform(RandomMaskFunc([0.08], [1])))
    print(len(dataset))
    ks, mask, y, x, mean, std, _, _, _ = dataset[420]

    import pylab as plt
    fig, ax = plt.subplots(nrows=2, figsize=(5, 4 * 2),
                           subplot_kw=dict(frameon=False, xticks=[], yticks=[]),
                           gridspec_kw=dict(wspace=0.0, hspace=0.0))
    ax[0].imshow(y[0])
    ax[1].imshow(x[0])
    plt.show()
    print(ks.shape, x.shape)
    print(ks.mean(), ks.std(), ks.max(), ks.min())
    print(x.mean(), x.std(), x.max(), x.min())
    print(np.linalg.norm(y.numpy() - x.numpy()))

