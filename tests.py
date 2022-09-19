# import unittest
import torch
import torch.utils.data
import pylab as plt
from optimize_pattern import mrisensesim, dice_loss
from optimize_pattern import PatternSampler
from optimize_pattern import LOUPE, LOUPErc, IGS
from optimize_pattern import ACDCDataset, BraTS2dDataset
from k_space_reconstruction.nets.unet import Unet

from tqdm import tqdm


def eval_image(data, model):
    pm = model(data['img'])
    gt = data['mask'].long()
    return dice_loss(gt, pm)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset = ACDCDataset('/home/a_razumov/small_datasets/acdc_seg_h5/train.h5', device)
    val_dataset = ACDCDataset('/home/a_razumov/small_datasets/acdc_seg_h5/val.h5', device)

    # train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(1,128).numpy().tolist())
    # val_dataset = torch.utils.data.Subset(train_dataset, torch.arange(1,32).numpy().tolist())

    train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Unet(1, 4).to(device).train(False).eval()
    for p in model.parameters():
        p.requires_grad = False
    model.load_state_dict(torch.load('./models/unet-acdc-norot.pt'))

    poptimizer = LOUPE(acceleration=1 / 16, img_shape=[256, 256], dimensions=1, device=device)

    ksampler = PatternSampler(ncoils=1, coil_width=32, shape=(256, 256), device=device)
    zm = torch.ones(256)
    zm[::2] = 0

    pbar = tqdm(range(100))
    for epoch in pbar:
        for data in train_generator:
            poptimizer.update_on_batch(f_func=lambda x: eval_image(ksampler(data, sampling=x), model))
        val_loss = 0
        for data in val_generator:
            with torch.no_grad():
                z = poptimizer.get_train_pattern()
                val_loss += eval_image(ksampler(data, sampling=z), model)
        pbar.set_description(f'loss: {val_loss.item()}')
        poptimizer.update_on_epoch()
