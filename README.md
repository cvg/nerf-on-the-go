<div align="center">

# NeRF <em>On-the-go</em>: Exploiting Uncertainty for Distractor-free NeRFs in the Wild

  <p align="center">
    <a href="https://github.com/rwn17"><strong>Weining Ren*</strong></a>
    ·
    <a href="https://zzh2000.github.io"><strong>Zihan Zhu*</strong></a>
    ·
    <a href="https://inf.ethz.ch/people/people-atoz/person-detail.MjY0ODc2.TGlzdC8zMDQsLTIxNDE4MTU0NjA=.html"><strong>Boyang Sun</strong></a>
    ·
    <a href="https://inf.ethz.ch/people/people-atoz/person-detail.Mjc4NTY0.TGlzdC8zMDQsLTIxNDE4MTU0NjA=.html"><strong>Julia Chen</strong></a>
    ·
    <a href="https://people.inf.ethz.ch/pomarc/"><strong>Marc Pollefeys</strong></a>
    ·
    <a href="https://pengsongyou.github.io"><strong>Songyou Peng</strong></a>
  </p>
  <p align="center"><strong>(* Equal Contribution)</strong></p>
  <h2 align="center">CVPR 2024</h2>
  <h3 align="center"><a href="https://arxiv.org/abs/2405.18715">Paper</a> | <a href="https://youtu.be/mUQ_LOyonB0?si=uacxom-6ur7_oGRw">Video</a> | <a href="https://nerf-on-the-go.github.io/">Project Page</a></h3>
  <p align="center">
  <a href="">
    <img src="./media/teaser.gif" alt="Logo" width="80%">
  </a>
</p>
  <div align="center"></div>

</div>

<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Description">Description</a>
    </li>
    <li>
      <a href="#Setup">Setup</a>
    </li>
    <li>
      <a href="#Dataset Preparation">Dataset Preparation</a>
    </li>
    <li>
      <a href="#Running">Running</a>
    </li>
    <li>
      <a href="#Citation">Citation</a>
    </li>
    <li>
      <a href="#Contact">Contact</a>
    </li>
  </ol>
</details>


## Description

This repository hosts the official Jax implementation of the paper "NeRF <em>on-the-go</em>: Exploiting Uncertainty for Distractor-free NeRFs in the Wild" (CVPR 2024). For more details, please visit our [project webpage](https://nerf-on-the-go.github.io/).

This Repo is built upon [Multinerf](https://github.com/google-research/multinerf) codebase.

## Setup

```
# Clone the repo.
git clone https://github.com/cvg/nerf-on-the-go
cd nerf-on-the-go

# Make a conda environment.
conda create --name on-the-go python=3.9
conda activate on-the-go

# Prepare pip.
conda install pip
pip install --upgrade pip


# Install requirements.
pip install -r requirements.txt

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap

# Confirm that all the unit tests pass.
./scripts/run_all_unit_tests.sh
```
You'll also need to update your [JAX](https://jax.readthedocs.io/en/latest/installation.html) installation to support GPUs or TPUs.

```
pip install  jax==0.4.26 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install  jaxlib==0.4.26+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Instructions for ETH Euler
<details>
  <summary>Click to expand</summary>

on ETH Euler, to support for GPU jax, you need to apply for a debug mode gpu and then upgrade the gcc and cuda
```
srun -n 4 --mem-per-cpu=12000 --gpus=rtx_3090:1 --gres=gpumem:20g --time=4:00:00 --pty bash
conda activate on-the-go
module load eth_proxy gcc/8.2.0 cuda/12.1.1 cudnn/8.9.2.26
```

After loading the modules, verify their activation by executing ```module list```. Occasionally, modules may not load correctly, requiring you to load each one individually. Following this, proceed with the Jax installation:

```
# Installs the wheel compatible with CUDA 12 and cuDNN 8.9 or newer.
pip install  jax==0.4.26 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install  jaxlib==0.4.26+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

After successful installation, please rerun ```./scripts/run_all_unit_tests.sh```.

The installation process outlined above has been verified on the Euler system using an RTX 3090. You may get a warning 
```
The NVIDIA driver's CUDA version is 12.0 which is older than the ptxas CUDA version (12.4.131). Because the driver is older than the ptxas version, XLA is disabling parallel compilation, which may slow down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.
``` 

But it's fine. The Euler supports up to CUDA 12.1, while JAX now requires a minimum of CUDA 12.3. As discussed in the JAX [Issue #18032](https://github.com/google/jax/issues/18032), this discrepancy primarily impacts compilation speed rather than overall functionality.
</details>


## Dataset Preparation

### Downloading the Dataset
To download the "On-the-go" dataset, execute the following command:
```bash
bash ./scripts/download_on-the-go.sh
```
This script not only downloads the dataset but also downsamples the images as required.

### Feature Extraction with DINOv2
For extracting features using the DINOv2, use the command below:
```bash
bash ./scripts/feature_extract.sh
```


After feature extraction, the dataset should be organized as 
```
on-the-go
├── arcdetriomphe
│   ├── images
│   ├── images_{DOWNSAMPLE_RATE}
│   ├── features_{DOWNSAMPLE_RATE}
│   ├── split.json
│   ├── transforms.json
├── ....
│
└── tree
    ├── images_{DOWNSAMPLE_RATE}
    ├── ....
    └── transforms.json
```

### Dataset Structure and Configuration Files
- **split.json**: This file outlines the train and evaluation splits, following the naming conventions used in the RobustNeRF dataset, categorized as 'clutter' and 'clean'.
- **transforms.json**: Contains pose and intrinsic information, formatted according to the Blender dataset format, derived from COLMAP files. Refer to the [Instant-NGP script](https://github.com/NVlabs/instant-ngp/blob/de507662d4b3398163e426fd426d48ff8f2895f6/scripts/colmap2nerf.py) for more details.

### Future Updates
We plan to expand support to include custom datasets in future updates.


## Running

Example scripts for training, evaluating, and rendering can be found in
`scripts/`. You'll need to change the paths to point to wherever the datasets
are located. [Gin](https://github.com/google/gin-config) configuration files
for our model and some ablations can be found in `configs/`.

1. Training on-the-go:
```
bash scripts/train_on-the-go.sh 
```

2. Evaluating on-the-go:
```
bash scripts/eval_on-the-go.sh
```

3. Rendering on-the-go:
```
bash scirpts/render_on-the-go.sh
```

Tensorboard is supported for logging.

### Note
Since we use a different recording device for ***arc de triomphe*** and ***patio*** scene, the image downsample rate(4 instead of 8) and feature downsample rate(2 instead of 4) is different. Please use a separate script to train them by 

```
bash scripts/train_on-the-go_HD.sh
```

### OOM errors

About **80G gpu memory** is needed to run current version.You may need to reduce the batch size (`Config.batch_size`) to avoid out of memory
errors. If you do this, but want to preserve quality, be sure to increase the number
of training iterations and decrease the learning rate by whatever scale factor you
decrease batch size by. 


## Todo

- [ ] Custom dataset tutorial
- [ ] Support LPIPS calculation
 
## Citation

If you use NeRF on-the-go, please cite 

```
@InProceedings{Ren2024NeRF,
    title={NeRF on-the-go: Exploiting Uncertainty for Distractor-free NeRFs in the Wild},
    author={Ren, Weining and Zhu, Zihan and Sun, Boyang and Chen, Jiaqi and Pollefeys, Marc and Peng, Songyou},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
```

Also, this code is built upon multinerf, feel free to cite this entire codebase as:

```
@misc{multinerf2022,
      title={{MultiNeRF}: {A} {Code} {Release} for {Mip-NeRF} 360, {Ref-NeRF}, and {RawNeRF}},
      author={Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman and Ricardo Martin-Brualla and Jonathan T. Barron},
      year={2022},
      url={https://github.com/google-research/multinerf},
}
```

## Contact
If there is any problem, please contact Weining by ren.weining@connect.hku.hk
