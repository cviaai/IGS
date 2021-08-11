[![License](https://img.shields.io/github/license/analysiscenter/pydens.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://python.org)

# Iterative Gradient Sampling (IGS)

This repository provides the official PyTorch implementation of Iterative Gradient Sampling (IGS) methods that introduces approach to segmentation and classification which dramatically reduces k-space fractions and introduces a novel method for pathology diagnostics.

<p align="center">
<img src="misc/igs_undersampling.gif" alt>

</p>
<p align="center">
<em>Demonstration of k-space patterning via IGS algorithm featuring segmentation results with the ACDC dataset.</em>
</p>


## Abstract
To accelerate MRI, the field of compressed sensing is traditionally concerned with optimizing the image quality after a partial undersampling of the measurable *k*-space. In our work, we propose to change the focus from the quality of the reconstructed image to the quality of the downstream image analysis outcome. 
Specifically, we propose to optimize the patterns according to how well a sought-after pathology could be detected or localized in the reconstructed images. 
We find the optimal undersampling patterns in *k*-space that maximize target value functions of interest in commonplace medical vision problems (reconstruction, segmentation, and classification) and propose a new iterative gradient sampling routine universally suitable for these tasks. 
We validate the proposed MRI acceleration paradigm on three classical medical datasets, demonstrating a noticeable improvement of the target metrics at the high acceleration factors (for the segmentation problem at x16 acceleration, we report up to 12% improvement in Dice score over the other undersampling patterns

## Usage

### 0. Cloning the repository

```bash
$ git clone https://github.com/cviaai/IGS.git
$ cd IGS-SEGMENT/
```

### 1. Creating python environment

```bash
$ conda env create -f igs.yml
$ conda activate igs
```

### 2. Datasets

IGS is trained to work with the major public medical datasets to resolve the following medical tasks:
1. Segmentation (BraTS2020 for brain tumor segmentation, ACDC for cardiac segmentation)
2. Classification (BraTS2020 total amount of slices being split into tumor/not tumor classes)
3. Image reconstruction (raw k-space from FastMRI dataset, undersampled ACDC, BraTS)

### 3. Training

Training on BRATS dataset

```bash
$ python main.py --mode train --dataset BRATS --crop_size 256 --image_size 256 --c_dim 1 \
                 --image_dir data/brats/syn \
                 --sample_dir brats_syn_256_lambda0.1/samples \
                 --log_dir brats_syn_256_lambda0.1/logs \
                 --model_save_dir brats_syn_256_lambda0.1/models \
                 --result_dir brats_syn_256_lambda0.1/results \
                 --batch_size 8 --num_workers 4 --lambda_id 0.1 --num_iters 300000
```

### 4. Testing

Testing on BRATS dataset

```bash
$ python main.py --mode test_brats --dataset BRATS --crop_size 256 --image_size 256 --c_dim 1 \
                 --image_dir data/brats/syn \
                 --sample_dir brats_syn_256_lambda0.1/samples \
                 --log_dir brats_syn_256_lambda0.1/logs \
                 --model_save_dir brats_syn_256_lambda0.1/models \
                 --result_dir brats_syn_256_lambda0.1/results \
                 --batch_size 16 --num_workers 4 --lambda_id 0.1 --test_iters 300000
```

### 5. Testing using pretrained models

Loss functions used:

<img src="https://render.githubusercontent.com/render/math?math=L_{DICE}(Y, \hat{Y})=1-\frac{2 Y \hat{Y}%2B1}{Y%2B\hat{Y}%2B1}">

<img src="https://render.githubusercontent.com/render/math?math=L_{BCE} = -\sum_{i=1}^{C=2}Y_{i} log (\hat{Y}_{i} )">

<img src="https://render.githubusercontent.com/render/math?math=L_{L1}=|| X_{i} - \hat{X_{i}}||_{1}">


## Citation

Please cite this work as following:

```
@misc{razumov-IGS,
  author = {Artem Razumov and Oleg Rogov and Dmitry V. Dylov},
  title = {Smart mask learning for segmentation},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cviaai/IGS-SEGMENT}}
}
```
