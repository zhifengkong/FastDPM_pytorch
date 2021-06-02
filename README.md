# Official PyTorch implementation for "On Fast Sampling of Diffusion Probabilistic Models".
FastDPM generation on CIFAR-10, CelebA, and LSUN datasets. See paper via [this link](https://arxiv.org/abs/2106.00132).

# Pretrained models
Download checkpoints from [this link](https://heibox.uni-heidelberg.de/d/01207c3f6b8441779abf/) and [this link](https://drive.google.com/file/d/1R_H-fJYXSH79wfSKs9D-fuKQVan5L-GR/view?usp=sharing). Put them under ```checkpoints\ema_diffusion_${dataset_name}_model\model.ckpt```, where ```${dataset_name}``` is ```cifar10```, ```celeba64```, ```lsun_bedroom```, ```lsun_church```, or ```lsun_cat```.

# Usage
General command: ```python generate.py -ema -name ${dataset_name} -approxdiff ${approximate_diffusion_process} -kappa ${kappa} -S ${FastDPM_length} -schedule ${noise_level_schedule} -n ${number_to_generate} -bs ${batchsize} -gpu ${gpu_index}```
- ```${dataset_name}```: ```cifar10```, ```celeba64```, ```lsun_bedroom```, ```lsun_church```, or ```lsun_cat```
- ```${approximate_diffusion_process}```: ```VAR``` or ```STEP```
- ```${kappa}```: a real value between 0 and 1
- ```${FastDPM_length}```: an integer between 1 and 1000; 10, 20, 50, 100 used in paper.
- ```${noise_level_schedule}```: ```linear``` or ```quadratic```

## CIFAR-10
Below are commands to generate CIFAR-10 images.
- Standard DDPM generation: ```python generate.py -ema -name cifar10 -approxdiff STD -n 16 -bs 16```
- FastDPM generation (STEP + DDPM-rev): ```python generate.py -ema -name cifar10 -approxdiff STEP -kappa 1.0 -S 50 -schedule quadratic -n 16 -bs 16```
- FastDPM generation (STEP + DDIM-rev): ```python generate.py -ema -name cifar10 -approxdiff STEP -kappa 0.0 -S 50 -schedule quadratic -n 16 -bs 16```
- FastDPM generation (VAR + DDPM-rev): ```python generate.py -ema -name cifar10 -approxdiff VAR -kappa 1.0 -S 50 -schedule quadratic -n 16 -bs 16```
- FastDPM generation (VAR + DDIM-rev): ```python generate.py -ema -name cifar10 -approxdiff VAR -kappa 0.0 -S 50 -schedule quadratic -n 16 -bs 16```

## CelebA
Below are commands to generate CelebA images. 
- Standard DDPM generation: ```python generate.py -ema -name celeba64 -approxdiff STD -n 16 -bs 16```
- FastDPM generation (STEP + DDPM-rev): ```python generate.py -ema -name celeba64 -approxdiff STEP -kappa 1.0 -S 50 -schedule linear -n 16 -bs 16```
- FastDPM generation (STEP + DDIM-rev): ```python generate.py -ema -name celeba64 -approxdiff STEP -kappa 0.0 -S 50 -schedule linear -n 16 -bs 16```
- FastDPM generation (VAR + DDPM-rev): ```python generate.py -ema -name celeba64 -approxdiff VAR -kappa 1.0 -S 50 -schedule linear -n 16 -bs 16```
- FastDPM generation (VAR + DDIM-rev): ```python generate.py -ema -name celeba64 -approxdiff VAR -kappa 0.0 -S 50 -schedule linear -n 16 -bs 16```

## LSUN_bedroom
Below are commands to generate LSUN bedroom images. 
- Standard DDPM generation: ```python generate.py -ema -name lsun_bedroom -approxdiff STD -n 8 -bs 8```
- FastDPM generation (STEP + DDPM-rev): ```python generate.py -ema -name lsun_bedroom -approxdiff STEP -kappa 1.0 -S 50 -schedule linear -n 8 -bs 8```
- FastDPM generation (STEP + DDIM-rev): ```python generate.py -ema -name lsun_bedroom -approxdiff STEP -kappa 0.0 -S 50 -schedule linear -n 8 -bs 8```
- FastDPM generation (VAR + DDPM-rev): ```python generate.py -ema -name lsun_bedroom -approxdiff VAR -kappa 1.0 -S 50 -schedule linear -n 8 -bs 8```
- FastDPM generation (VAR + DDIM-rev): ```python generate.py -ema -name lsun_bedroom -approxdiff VAR -kappa 0.0 -S 50 -schedule linear -n 8 -bs 8```

## Note
To generate 50K samples, set ```-n 50000``` and batchsize (```-bs```) divisible by 50K. 

# Compute FID
To compute FID of generated samples, first make sure there are 50K images, and then run
- ```python FID.py -ema -name cifar10 -approxdiff STEP -kappa 1.0 -S 50 -schedule quadratic```

# Code References
- [DDPM TensorFlow official](https://github.com/hojonathanho/diffusion)
- [DDPM PyTorch](https://github.com/pesser/pytorch_diffusion)
- [DDPM CelebA-HQ](https://github.com/FengNiMa/pytorch_diffusion_model_celebahq)
- [DDIM PyTorch](https://github.com/ermongroup/ddim)
- [FID PyTorch](https://github.com/mseitzer/pytorch-fid)
- [DiffWave PyTorch 1](https://github.com/lmnt-com/diffwave)
- [DiffWave PyTorch 2](https://github.com/philsyn/DiffWave-Vocoder)
- [DiffWave PyTorch 3](https://github.com/philsyn/DiffWave-unconditional)