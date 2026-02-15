# FID-diffusers

The code in this repository can be used to calculate the FID scores of different diffusion pipelines based on custom schedulers.
- The solvers are all implemented in this [fork](https://github.com/vlcanesin/rk-diffusers). In order to install it, you can clone the repo and install the library in your environment using `pip install -e .`
- Once the library is installed, you'll be able to execute `evaluation.py`, the main evaluation script.

## Introduced Solvers

Checkout the [fork](https://github.com/vlcanesin/rk-diffusers) for more information.
  
## Requirements

1. Create a new virtual environment with the requirements listed in `requirements.txt` in your desired directory:
```
# cd my-environment-directory
python -m venv my-venv
source my-venv/bin/activate

# cd FID-diffusers
pip install -r requirements.txt
```

2. Select a model from the HuggingFace hub to use. The only ones that were tested and that are accepted by the evaluation script are:
- `google/ddpm-cifar10-32` : used for the 32x32 CIFAR10 dataset. A better model can be downloaded with [this link](https://github.com/VainF/Diff-Pruning/releases/download/v0.0.1/ddpm_ema_cifar10.zip), and the code currently supports this version.
- `google/ddpm-bedroom-256` : used for the 256x256 LSUN-bedroom dataset.

3. Download the dataset used to compute the FID score. Important: they must have at least 50k images for an accurate FID measure.
- For CIFAR10, this can be done directly using torchvision: 
```
torchvision.datasets.CIFAR10(root="/beegfs/vrosadac/datasets/cifar10", train=True, download=True)
```
- For LSUN-bedroom, the dataset was downloaded from [this link](https://www.kaggle.com/datasets/jhoward/lsun_bedroom) and the first 50k images were taken and sorted by using the `./utils/get_lsun.py` script (adjust `source_dir` and `dest_dir` as needed)

## Evaluation

The `evaluation.py` script runs distributedly across different GPUs and is designed to run with `torchrun`. It accepts a list of parameters:
1. `--solvers` : the solvers/schedulers to be evaluated. The following strings are allowed:
```
DDPM, DDIM, DPMSolver, DPMEDM, DPMComposed, DPMEDMComposed, DPMEDMComposed_high, RK1, RK2, RK3, RK4, RKEDM1, RKEDM2, RKEDM3, RKEDM, RKfEDM1, RKfEDM2, RKfEDM3, RKfEDM, RKcomp, RKEDMcomp, RKEDMcomp_high, RKfEDMcomp, RKfEDMcomp_high, pRK2, pRK3, pRK4, pRKEDM2, pRKEDM3, pRKEDM4, ExpRK4, ExpRK5, ExpRK4_mid, ExpRK4_simp, ExpRK5_mid, ExpRK5_simp, ExpRKEDM4, ExpRKEDM5, ExpRKEDM4_mid, ExpRKEDM4_simp, ExpRKEDM5_mid, ExpRKEDM5_simp
```
2. `--steps` : the number of denoising steps each solver will be evaluated on.
3. `--dataset` : the dataset (and consequently the model) used for the evaluation. Can be one of the following: `CIFAR10` or `LSUN-bedroom`
4. `--bsize` : the batch size used for each GPU
5. `--csvdir` : the directory of the CSV output file containing the statistics (defaults to `./csv`)
6. `--modelpath` : the path to the cached model
7. `--datapath` : the path to the cached dataset
8. `--model` : which model architecture to use (SimpleUNet, UNet2DModel - diffusers, DiT-S/4)

This is an example of a valid command:
```
torchrun --nnodes=1 --nproc-per-node=2 evaluation.py --solvers DDPM DDIM --steps 100 200 500 --dataset LSUN-bedroom --bsize 16 --modelpath ./ddpm-bedroom-256 --datapath ./lsun
```
