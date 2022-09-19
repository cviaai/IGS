# Take a look at strings where I've left '# HARDCODED DATA INPUT' comment, this is the places where
# we have to replace a way to input data.

# Возможно надо переработать код чтобы засунуть его в класс.
# Логика всей функции такова: мы берем заданное число слайсов из датасета. Рандомно сэмплим количество столбцов,
# которое каждый сайс будет вкладывать в финальный kspace, заносим в список. На основе этого списка делаем маску,
# со значениями от 0 до num_of_slices_per_artifact.
# Далее просто по очереди перемножаем все kspace-ы с маской и складываем все в финальный kspace

import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from k_space_reconstruction.utils.kspace import pt_spatial2kspace, pt_kspace2spatial
from k_space_reconstruction.datasets.fastmri import FastMRIh5Dataset, FastMRITransform, RandomMaskFunc

random.seed(19)
torch.manual_seed(228)
torch.cuda.manual_seed(228)
np.random.seed(228)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


def show_kspace(kspace_list):
    """
    Plot a list of kspaces of one kspace
    """
    n = len(kspace_list)

    fig, ax = plt.subplots(1, n, figsize=(30, 4 * 5),
                           subplot_kw=dict(frameon=False, xticks=[], yticks=[]),
                           gridspec_kw=dict(wspace=0.0, hspace=0.0))
    if n == 1:
        ax.imshow((kspace_list[0][0].abs() + 1e-11).log())
    else:
        for i in range(n):
            ax[i].imshow((kspace_list[i][0].abs() + 1e-11).log())


def shift_one_dim(x: torch.Tensor, shift: int, dim: int, direction: str, bg_shape=[10, 10]) -> torch.Tensor:
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    bg_sample = torch.zeros([bg_shape[0], bg_shape[1]])
    x_img = x[0]
    for i in range(bg_shape[0]):
        for j in range(bg_shape[1]):
            bg_sample[i][j] = x_img[i][j]
    bg_sample_means = torch.mean(bg_sample, dim=1)

    def generate_bg_value(samples):
        sample_std = torch.std(samples, unbiased=False)
        sample_shape = samples.shape

        num = random.choice(samples)
        val = np.random.uniform(num - 2 * sample_std, num + 2 * sample_std)
        return val

    def generate_padding(pad):
        for i in range(pad.shape[-2]):
            for j in range(pad.shape[-1]):
                pad[0][i][j] = generate_bg_value(bg_sample_means)
        return pad

    if direction == 'left':
        left = x.narrow(dim, shift, x.size(dim) - shift)

        pad = torch.zeros([x.shape[0], x.shape[1], x.shape[2] - left.shape[2]])
        pad = generate_padding(pad)
        shifted_img = torch.cat((left, pad), dim=dim)

    elif direction == 'right':
        right = x.narrow(dim, 0, x.size(dim) - shift)

        pad = torch.zeros([x.shape[0], x.shape[1], x.shape[2] - right.shape[2]])
        pad = generate_padding(pad)
        shifted_img = torch.cat((pad, right), dim=dim)

    elif direction == 'further':
        further = x.narrow(dim, 0, x.size(dim) - shift)

        pad = torch.zeros([x.shape[0], x.shape[1] - further.shape[1], x.shape[2]])
        pad = generate_padding(pad)
        shifted_img = torch.cat((pad, further), dim=dim)

    elif direction == 'closer':
        closer = x.narrow(dim, shift, x.size(dim) - shift)

        pad = torch.zeros([x.shape[0], x.shape[1] - closer.shape[1], x.shape[2]])
        pad = generate_padding(pad)
        shifted_img = torch.cat((closer, pad), dim=dim)

    return shifted_img


def sum_to_one(n):
    values = [0.0, 1.0] + [random.random() for _ in range(n - 1)]
    values.sort()

    def toFixed(numObj, digits=3):
        return float(f"{numObj:.{digits}f}")

    per_cent_list = [toFixed(values[i + 1] - values[i]) for i in range(n)]
    return per_cent_list


def calc_each_slice_contribution(sample_kspace, num_of_slices_per_artifact=4):
    """
    Returns a list with number of slice and randomly sampled
    amount of this slice contribution. Slice numeration starts from 0
    """
    total_row_num = sample_kspace.shape[-1]

    percent_of_each_slice = sum_to_one(num_of_slices_per_artifact)
    rows_of_each_slice = [int(total_row_num * p) for p in percent_of_each_slice]

    if sum(rows_of_each_slice) != total_row_num:
        rows_residue = total_row_num - sum(rows_of_each_slice[:-1])
        rows_of_each_slice = rows_of_each_slice[:-1]
        rows_of_each_slice.append(rows_residue)

    return rows_of_each_slice


def apply_mask_to_kspace(kspace_list, mask, dim):
    masked_kspace_list = []
    for idx, kspace in enumerate(kspace_list):
        aligned_mask = mask == idx
        masked_kspace_list.append(kspace[dim] * aligned_mask)
    final_kspace = torch.sum(torch.stack(masked_kspace_list), dim=0)
    return final_kspace


def get_sample_kspace(dataset: FastMRIh5Dataset, idx: int):
    _f, slice_id = dataset._slices[idx]
    ks = dataset.hf[_f][:]
    ks = ks * 1e6
    return torch.from_numpy(ks), slice_id


def add_motion_artefacts(dataset, item, motion_artefact_coefficient=0.99,
                         num_of_slices_per_artifact=4):
    """
    Physically we can move our body part in 3 dimensions.
    x axis: left/right, y axis: further/closer, z axis: top/bottom

    In theory we can move body part in all three dimentions simultaneously. For now function moves
    body part ONLY in one direction.
    Also, we have to choose PHISICALLY POSSIBLE range of movement for each axes.
    """
    prob = np.random.randint(100)
    if prob <= motion_artefact_coefficient * 100:

        full_kspace, slice_id = get_sample_kspace(dataset, item)
        full_kspace = torch.stack([full_kspace.real, full_kspace.imag], dim=1)
        sample_kspace = full_kspace[slice_id]
        rows_of_each_slice = calc_each_slice_contribution(sample_kspace, num_of_slices_per_artifact)
        motion_mask = torch.zeros([sample_kspace.shape[-1], sample_kspace.shape[-2]])

        direction_list = ['top', 'bottom',
                          'left', 'right',
                          'further', 'closer']
        direction = random.choice(direction_list)
        direction = 'further'
        print("Monvement Direction: ", direction)

        if direction == 'left' or direction == 'right':
            full_kspace, slice_id = get_sample_kspace(dataset, item)
            full_kspace = torch.stack([full_kspace.real, full_kspace.imag], dim=1)
            sample_kspace = full_kspace[slice_id]

            slice_img = abs(pt_kspace2spatial(sample_kspace[0] + 1j * sample_kspace[1]))
            slice_img = torch.unsqueeze(slice_img, 0)

            current_row = 0
            kspace_list = []
            for idx, pixel_num in enumerate(rows_of_each_slice):
                shift_val = random.randint(0, 20)  # WHAT NUMBER TO CHOOSE?
                slice_img = shift_one_dim(slice_img, shift_val, 2, direction=direction)
                slice_img_kspace = pt_spatial2kspace(slice_img)[0]

                slice_img_kspace = torch.stack((slice_img_kspace.real, slice_img_kspace.imag))
                kspace_list.append(slice_img_kspace)

                motion_mask[current_row:pixel_num + current_row] = idx
                current_row = current_row + pixel_num

        elif direction == 'further' or direction == 'closer':
            full_kspace, slice_id = get_sample_kspace(dataset, item)
            full_kspace = torch.stack([full_kspace.real, full_kspace.imag], dim=1)
            sample_kspace = full_kspace[slice_id]

            slice_img = abs(pt_kspace2spatial(sample_kspace[0] + 1j * sample_kspace[1]))
            slice_img = torch.unsqueeze(slice_img, 0)

            current_row = 0
            kspace_list = []
            for idx, pixel_num in enumerate(rows_of_each_slice):
                shift_val = random.randint(0, 20)  # WHAT NUMBER TO CHOOSE?
                slice_img = shift_one_dim(slice_img, shift_val, 1, direction=direction)
                slice_img_kspace = pt_spatial2kspace(slice_img)[0]

                slice_img_kspace = torch.stack((slice_img_kspace.real, slice_img_kspace.imag))
                kspace_list.append(slice_img_kspace)

                motion_mask[current_row:pixel_num + current_row] = idx
                current_row = current_row + pixel_num

        elif direction == 'top' or direction == 'bottom':
            full_kspace, slice_id = get_sample_kspace(dataset, item)
            full_kspace = torch.stack([full_kspace.real, full_kspace.imag], dim=1)
            kspace_list = [full_kspace[(slice_id + i) % full_kspace.shape[0]] for i in range(num_of_slices_per_artifact)]    # HARDCODED DATA INPUT

            if direction == 'bottom':
                kspace_list = kspace_list[::-1]

            current_row = 0
            for idx, pixel_num in enumerate(rows_of_each_slice[:-1]):
                motion_mask[current_row:pixel_num + current_row] = idx + 1
                current_row = current_row + pixel_num
        else:
            raise ValueError

        t_motion_mask = np.array([[motion_mask[j][i] for j in range(len(motion_mask))]
                                  for i in range(len(motion_mask[0]))])
        kspace = torch.stack((apply_mask_to_kspace(kspace_list, t_motion_mask, dim=0),
                              apply_mask_to_kspace(kspace_list, t_motion_mask, dim=1)))
        return kspace


def main():
    dir_train = '/run/media/airplaneless/Backup/DATA/fastMRIh5/singlecoil_train.h5'
    dataset = FastMRIh5Dataset(dir_train, FastMRITransform(RandomMaskFunc([0.08], [1])))
    kspace = add_motion_artefacts(dataset, item=135, num_of_slices_per_artifact=5)  # num_of_slices_per_artifact WHAT NUMBER TO CHOOSE?

    k = kspace[0] + 1j * kspace[1]
    img = pt_kspace2spatial(k)

    print(kspace.shape)
    fig, ax = plt.subplots(2, 1, figsize=(4, 10))
    ax[0].imshow(abs(img))
    ax[1].imshow((kspace[0].abs() + 1e-11).log())
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
