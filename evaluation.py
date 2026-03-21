import os

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from diffusers import (
    UNet2DModel, 
    RungeKuttaScheduler, 
    DDPMScheduler, 
    DDIMScheduler, 
    DPMSolverSinglestepScheduler, 
    WiSPRungeKuttaScheduler, 
    ExpRungeKuttaScheduler,
    DPMSolverOSchScheduler
)

import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance

from utils.torch_timer import Timer
from pipelines import CustomDiffusionPipeline, RungeKuttaPipeline

import torchvision

import csv
from datetime import datetime

import argparse

# Additional models (don't use the huggingface interface)
from models.SimpleUNet import SimpleUNet
from models.DiT import DiT_models

def load_model(args, device):
    model_name = args.model
    path = args.modelpath

    if model_name == "unet_diffusers":
        if args.dataset == "CIFAR10":
            if path == "default":
                path = "/beegfs/vrosadac/models/cifar10-ema-model-790000/pretrained/ddpm_ema_cifar10" 
            model = UNet2DModel.from_pretrained(path, subfolder="unet")

        elif args.dataset == "LSUN-bedroom":
            if path == "default":
                path = "google/ddpm-bedroom-256"
            model = UNet2DModel.from_pretrained(path)

    elif model_name == "unet_simple":
        model = SimpleUNet()

        if path == "default":
            path = "models/SimpleUNet-final.pt"

        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    elif model_name == "dit_cifar": 
        model = DiT_models["DiT-S/4"](
            input_size=32,
            num_classes=0
        )

        if path == "default":
            path = "models/DiT-0280000.pt"

        torch.serialization.add_safe_globals([argparse.Namespace])
        ckpt = torch.load(path, map_location="cpu")
        if "ema" in ckpt:
            ckpt = ckpt["ema"]
        model.load_state_dict(ckpt)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.eval().to(device)

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def prepare_for_fid(x):
    # x: (B, 3, H, W), values in [0, 1]
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    x = (x * 255).clamp(0, 255).to(torch.uint8)
    return x

def gather_batch(tensor: torch.Tensor, local_rank) -> torch.Tensor:
    """Gather a tensor from all processes. Returns the concatenated tensor."""
    world_size = dist.get_world_size()
    if local_rank == 0:
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    else:
        gathered = None
    dist.gather(tensor, gathered, dst=0)
    if local_rank == 0:
        return torch.cat(gathered, dim=0)
    else:
        return None

# Calculate stats for one pipeline
def calculate_stats(dataloader, image_shape, pipeline, num_inference_steps, local_rank):
    device = torch.device(f"cuda:{local_rank}")
    generator = torch.Generator(device=device)
    generator.manual_seed(42 + local_rank)

    data_iter = None
    if local_rank == 0:
        fid_metric = FrechetInceptionDistance(feature=2048, dist_sync_on_step=False, sync_on_compute=False).to(device)
        data_iter = iter(dataloader)

    total_inference_time = 0.0
    total_images = 0

    torch.cuda.reset_peak_memory_stats()

    has_data = torch.tensor(0, device=device)
    local_bsz = torch.tensor(0, device=device)
    while True:
        # Computes FID for real images and broadcasts batch size
        if local_rank == 0:
            try:
                real_images, _ = next(data_iter)
                has_data = torch.tensor(1, device=device)

                fid_metric.update(prepare_for_fid(real_images.to(device)), real=True)

                total_bsz = torch.tensor(real_images.size(0), device=device)
                local_bsz = total_bsz // dist.get_world_size()
            except StopIteration:
                has_data = torch.tensor(0, device=device)

        dist.broadcast(has_data, src=0)
        dist.broadcast(local_bsz, src=0)

        if has_data.item() == 0:
            break

        # Generate images
        batch_shape = (local_bsz.item(),) + image_shape
        with Timer(use_gpu=True) as t:
            generated_images = pipeline(batch_shape, num_inference_steps=num_inference_steps, generator=generator)
        total_inference_time += t.elapsed
        total_images += local_bsz.item()

        # Normalize to [0, 1]
        generated_images = (generated_images * 0.5 + 0.5).clamp(0, 1)

        # Gather across GPUs
        all_fake = gather_batch(generated_images, local_rank)

        # Only rank 0 updates FID
        if local_rank == 0:
            fid_metric.update(prepare_for_fid(all_fake.to(device)), real=False)

    # Reduce timing and memory
    total_time_tensor = torch.tensor(total_inference_time, device=device)
    total_images_tensor = torch.tensor(total_images, device=device)
    dist.all_reduce(total_time_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_images_tensor, op=dist.ReduceOp.SUM)

    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    print(f"(Device {local_rank}) Peak memory :", peak_memory)

    peak_memory_tensor = torch.tensor(peak_memory, device=device)
    dist.all_reduce(peak_memory_tensor, op=dist.ReduceOp.MAX)

    # Rank 0 computes final FID
    if local_rank == 0:
        fid_score = fid_metric.compute().item()
        mean_time = total_time_tensor.item() / total_images_tensor.item()
        stats = {
            "fid": fid_score,
            "inference-time": mean_time,
            "peak-gpu-memory": peak_memory_tensor.item()
        }
    else:
        stats = None

    return stats


def nfe_to_steps(solver_name, nfe, solver_obj=None):
    """
    Convert NFE to num_inference_steps, using predefined factors
    and solver-specific methods if available.
    
    solver_name: str, e.g., "DPMOSch", "RK3", "DDIM", etc.
    nfe: int, number of inference nfe
    solver_obj: the instantiated scheduler/solver object (optional)
    """
    nfe_factor = {
        "DDPM": 1,
        "DDIM": 1,
        "DPMSolver": 1,
        "DPMEDM": 1,
        "DPMOSch": None,
        "DPMEDMOSch": None,
        "DPMEDMOSch_high": None,
        "RK1": 1,
        "RK2": 1/2,
        "RK3": 1/3,
        "RK4": 1/4,
        "RKEDM1": 1,
        "RKEDM2": 1/2,
        "RKEDM3": 1/3,
        "RKEDM": 1/4,
        "RKfEDM1": 1,
        "RKfEDM2": 1/2,
        "RKfEDM3": 1/3,
        "RKfEDM": 1/4,
        "RKEDMOSch": None,
        "RKEDMOSch_high": None,
        "RKOSch": None,
        "RKfEDMcomp": None,
        "RKfEDMcomp_high": None,
        "WiSPRK2": 1,
        "WiSPRK3": 1,
        "WiSPRK4": 1,
        "WiSPRKEDM2": 1,
        "WiSPRKEDM3": 1,
        "WiSPRKEDM4": 1,
        "ExpRK4": 1/4,
        "ExpRK5": 1/5,
        "ExpRK4_mid": 1/4,
        "ExpRK4_simp": 1/4,
        "ExpRK5_mid": 1/5,
        "ExpRK5_simp": 1/5,
        "ExpRKEDM4": 1/4,
        "ExpRKEDM5": 1/5,
        "ExpRKEDM4_mid": 1/4,
        "ExpRKEDM4_simp": 1/4,
        "ExpRKEDM5_mid": 1/5,
        "ExpRKEDM5_simp": 1/5
    }

    factor = nfe_factor.get(solver_name)
    
    if factor is not None:
        return int(nfe * factor)

    # If the solver object has _nfe_to_steps, call it
    if solver_obj is not None and hasattr(solver_obj, "_nfe_to_steps"):
        return solver_obj._nfe_to_steps(nfe)
    
    raise ValueError(f"Cannot convert steps to NFE for solver {solver_name}")


def steps_to_nfe(solver_name, steps, solver_obj=None):
    """
    Convert num_inference_steps to NFE, using predefined factors
    and solver-specific methods if available.
    
    solver_name: str, e.g., "DPMOSch", "RK3", "DDIM", etc.
    steps: int, number of inference steps
    solver_obj: the instantiated scheduler/solver object (optional)
    """
    nfe_factor = {
        "DDPM": 1,
        "DDIM": 1,
        "DPMSolver": 1,
        "DPMEDM": 1,
        "DPMOSch": None,
        "DPMEDMOSch": None,
        "DPMEDMOSch_high": None,
        "RK1": 1,
        "RK2": 2,
        "RK3": 3,
        "RK4": 4,
        "RKEDM1": 1,
        "RKEDM2": 2,
        "RKEDM3": 3,
        "RKEDM": 4,
        "RKfEDM1": 1,
        "RKfEDM2": 2,
        "RKfEDM3": 3,
        "RKfEDM": 4,
        "RKEDMOSch": None,
        "RKEDMOSch_high": None,
        "RKOSch": None,
        "RKfEDMcomp": None,
        "RKfEDMcomp_high": None,
        "WiSPRK2": 1,
        "WiSPRK3": 1,
        "WiSPRK4": 1,
        "WiSPRKEDM2": 1,
        "WiSPRKEDM3": 1,
        "WiSPRKEDM4": 1,
        "ExpRK4": 4,
        "ExpRK5": 5,
        "ExpRK4_mid": 4,
        "ExpRK4_simp": 4,
        "ExpRK5_mid": 5,
        "ExpRK5_simp": 5,
        "ExpRKEDM4": 4,
        "ExpRKEDM5": 5,
        "ExpRKEDM4_mid": 4,
        "ExpRKEDM4_simp": 4,
        "ExpRKEDM5_mid": 5,
        "ExpRKEDM5_simp": 5
    }

    factor = nfe_factor.get(solver_name)
    
    if factor is not None:
        return int(steps * factor)

    # If the solver object has _steps_to_nfe, call it
    if solver_obj is not None and hasattr(solver_obj, "_steps_to_nfe"):
        return solver_obj._steps_to_nfe(steps)
    
    raise ValueError(f"Cannot convert steps to NFE for solver {solver_name}")


if __name__ == "__main__":

    script_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description="Parameters used for controling the tests")

    parser.add_argument('--solvers', type=str, nargs='+', help="List of solvers", required=True)
    parser.add_argument('--nfes', type=int, nargs='+', help="List of NFEs", required=True)
    parser.add_argument('--dataset', type=str, help="Name of the dataset used", required=True)
    parser.add_argument('--bsize', type=int, help="Batch size (per GPU)", required=True)
    parser.add_argument('--csvdir', type=str, help="CSV output directory", default="./csv")
    parser.add_argument('--datapath', type=str, help="Dataset path for FID calculation", required=True)
    
    parser.add_argument(
        "--model",
        type=str,
        default="unet_diffusers",
        choices=["unet_diffusers", "unet_simple", "dit_cifar"],
        help="Which model architecture to use"
    )

    parser.add_argument(
        "--modelpath",
        type=str,
        default="default",
        help="Checkpoint or pretrained model path"
    )

    args = parser.parse_args()

    assert args.dataset in ["CIFAR10", "LSUN-bedroom"], "Invalid dataset. Choose one of the following: CIFAR10 or LSUN-bedroom"

    local_rank = setup_distributed()

    # Load components
    if local_rank == 0:
        print("Loading components")

    device = torch.device(f"cuda:{local_rank}")
    eps_model = load_model(args, device)

    DDPM = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    DDIM = DDIMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    
    DPMSolver = DPMSolverSinglestepScheduler(
        num_train_timesteps=1000, prediction_type="epsilon",
        solver_order=3, algorithm_type="dpmsolver", final_sigmas_type="sigma_min",
        lower_order_final=True
    )
    DPMEDM = DPMSolverSinglestepScheduler(
        num_train_timesteps=1000, prediction_type="epsilon",
        solver_order=3, algorithm_type="dpmsolver", final_sigmas_type="sigma_min",
        lower_order_final=True, use_karras_sigmas=True
    )
    
    DPMOSch = DPMSolverOSchScheduler(
        num_train_timesteps=1000, prediction_type="epsilon",
        solver_order=3, algorithm_type="dpmsolver", final_sigmas_type="sigma_min",
        lower_order_final=True
    )

    DPMEDMOSch = DPMSolverOSchScheduler(
        num_train_timesteps=1000, prediction_type="epsilon",
        solver_order=3, algorithm_type="dpmsolver", final_sigmas_type="sigma_min",
        lower_order_final=True, use_karras_sigmas=True
    )
    DPMEDMOSch.set_scheduling_mode("edm")

    DPMEDMOSch_high = DPMSolverOSchScheduler(
        num_train_timesteps=1000, prediction_type="epsilon",
        solver_order=3, algorithm_type="dpmsolver", final_sigmas_type="sigma_min",
        lower_order_final=True, use_karras_sigmas=True
    )
    DPMEDMOSch_high.set_scheduling_mode("edm-high")

    rk1 = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="linear", order=1, use_order_scheduling=False)
    rk2 = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="linear", order=2, use_order_scheduling=False)
    rk3 = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="linear", order=3, use_order_scheduling=False)
    rk4 = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="linear", order=4, use_order_scheduling=False)

    RKEDM1 = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="edm", order=1, use_order_scheduling=False)
    RKEDM2 = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="edm", order=2, use_order_scheduling=False)
    RKEDM3 = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="edm", order=3, use_order_scheduling=False)
    RKEDM = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="edm", order=4, use_order_scheduling=False)

    RKfEDM1 = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="fake-edm-original", order=1, use_order_scheduling=False)
    RKfEDM2 = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="fake-edm-original", order=2, use_order_scheduling=False)
    RKfEDM3 = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="fake-edm-original", order=3, use_order_scheduling=False)
    RKfEDM = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="fake-edm-original", order=4, use_order_scheduling=False)

    RKOSch = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="linear", use_order_scheduling=True) 
    RKEDMOSch = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="edm", use_order_scheduling=True)
    RKEDMOSch_high = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="edm-high", use_order_scheduling=True)

    RKfEDMcomp = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="fake-edm-original", use_order_scheduling=True)
    RKfEDMcomp_high = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="fake-edm-high", use_order_scheduling=True)

    WiSPRK2 = WiSPRungeKuttaScheduler(num_train_timesteps=1000, order=2)
    WiSPRK3 = WiSPRungeKuttaScheduler(num_train_timesteps=1000, order=3)
    WiSPRK4 = WiSPRungeKuttaScheduler(num_train_timesteps=1000, order=4)
    WiSPRKEDM2 = WiSPRungeKuttaScheduler(num_train_timesteps=1000, order=2, timestep_schedule="edm")
    WiSPRKEDM3 = WiSPRungeKuttaScheduler(num_train_timesteps=1000, order=3, timestep_schedule="edm")
    WiSPRKEDM4 = WiSPRungeKuttaScheduler(num_train_timesteps=1000, order=4, timestep_schedule="edm")

    ExpRK4 = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=4)
    ExpRK5 = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=5)
    ExpRK4_mid  = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=4, quadrature="midpoint")
    ExpRK4_simp = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=4, quadrature="simpson")
    ExpRK5_mid  = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=5, quadrature="midpoint")
    ExpRK5_simp = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=5, quadrature="simpson")

    ExpRKEDM4 = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=4, timestep_schedule="edm")
    ExpRKEDM5 = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=5, timestep_schedule="edm")
    ExpRKEDM4_mid  = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=4, quadrature="midpoint", timestep_schedule="edm")
    ExpRKEDM4_simp = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=4, quadrature="simpson", timestep_schedule="edm")
    ExpRKEDM5_mid  = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=5, quadrature="midpoint", timestep_schedule="edm")
    ExpRKEDM5_simp = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=5, quadrature="simpson", timestep_schedule="edm")

    # Create the pipelines
    if local_rank == 0:
        print("Creating pipelines")
    DDPM_pipe = CustomDiffusionPipeline(model=eps_model, scheduler=DDPM).to("cuda", local_rank)
    DDIM_pipe = CustomDiffusionPipeline(model=eps_model, scheduler=DDIM).to("cuda", local_rank)
    
    DPMSolver_pipe = CustomDiffusionPipeline(model=eps_model, scheduler=DPMSolver).to("cuda", local_rank)
    DPMEDM_pipe = CustomDiffusionPipeline(model=eps_model, scheduler=DPMEDM).to("cuda", local_rank)
    DPMOSch_pipe = CustomDiffusionPipeline(model=eps_model, scheduler=DPMOSch).to("cuda", local_rank) 
    DPMEDMOSch_pipe = CustomDiffusionPipeline(model=eps_model, scheduler=DPMEDMOSch).to("cuda", local_rank)
    DPMEDMOSch_high_pipe = CustomDiffusionPipeline(model=eps_model, scheduler=DPMEDMOSch_high).to("cuda", local_rank)

    rk1_pipe = RungeKuttaPipeline(model=eps_model, scheduler=rk1).to("cuda", local_rank)
    rk2_pipe = RungeKuttaPipeline(model=eps_model, scheduler=rk2).to("cuda", local_rank)
    rk3_pipe = RungeKuttaPipeline(model=eps_model, scheduler=rk3).to("cuda", local_rank)
    rk4_pipe = RungeKuttaPipeline(model=eps_model, scheduler=rk4).to("cuda", local_rank)

    RKEDM1_pipe = RungeKuttaPipeline(model=eps_model, scheduler=RKEDM1).to("cuda", local_rank)
    RKEDM2_pipe = RungeKuttaPipeline(model=eps_model, scheduler=RKEDM2).to("cuda", local_rank)
    RKEDM3_pipe = RungeKuttaPipeline(model=eps_model, scheduler=RKEDM3).to("cuda", local_rank)
    RKEDM_pipe = RungeKuttaPipeline(model=eps_model, scheduler=RKEDM).to("cuda", local_rank)
    RKfEDM1_pipe = RungeKuttaPipeline(model=eps_model, scheduler=RKfEDM1).to("cuda", local_rank)
    RKfEDM2_pipe = RungeKuttaPipeline(model=eps_model, scheduler=RKfEDM2).to("cuda", local_rank)
    RKfEDM3_pipe = RungeKuttaPipeline(model=eps_model, scheduler=RKfEDM3).to("cuda", local_rank)
    RKfEDM_pipe = RungeKuttaPipeline(model=eps_model, scheduler=RKfEDM).to("cuda", local_rank)

    RKOSch_pipe = RungeKuttaPipeline(model=eps_model, scheduler=RKOSch).to("cuda", local_rank)
    RKEDMOSch_pipe = RungeKuttaPipeline(model=eps_model, scheduler=RKEDMOSch).to("cuda", local_rank)
    RKEDMOSch_high_pipe = RungeKuttaPipeline(model=eps_model, scheduler=RKEDMOSch_high).to("cuda", local_rank)

    RKfEDMcomp_pipe = RungeKuttaPipeline(model=eps_model, scheduler=RKfEDMcomp).to("cuda", local_rank)
    RKfEDMcomp_high_pipe = RungeKuttaPipeline(model=eps_model, scheduler=RKfEDMcomp_high).to("cuda", local_rank)

    WiSPRK2_pipe = RungeKuttaPipeline(model=eps_model, scheduler=WiSPRK2).to("cuda", local_rank)
    WiSPRK3_pipe = RungeKuttaPipeline(model=eps_model, scheduler=WiSPRK3).to("cuda", local_rank)
    WiSPRK4_pipe = RungeKuttaPipeline(model=eps_model, scheduler=WiSPRK4).to("cuda", local_rank)
    WiSPRKEDM2_pipe = RungeKuttaPipeline(model=eps_model, scheduler=WiSPRKEDM2).to("cuda", local_rank)
    WiSPRKEDM3_pipe = RungeKuttaPipeline(model=eps_model, scheduler=WiSPRKEDM3).to("cuda", local_rank)
    WiSPRKEDM4_pipe = RungeKuttaPipeline(model=eps_model, scheduler=WiSPRKEDM4).to("cuda", local_rank)

    ExpRK4_pipe = RungeKuttaPipeline(model=eps_model, scheduler=ExpRK4).to("cuda", local_rank)
    ExpRK5_pipe = RungeKuttaPipeline(model=eps_model, scheduler=ExpRK5).to("cuda", local_rank)
    ExpRK4_mid_pipe  = RungeKuttaPipeline(model=eps_model, scheduler=ExpRK4_mid).to("cuda", local_rank)
    ExpRK4_simp_pipe = RungeKuttaPipeline(model=eps_model, scheduler=ExpRK4_simp).to("cuda", local_rank)
    ExpRK5_mid_pipe  = RungeKuttaPipeline(model=eps_model, scheduler=ExpRK5_mid).to("cuda", local_rank)
    ExpRK5_simp_pipe = RungeKuttaPipeline(model=eps_model, scheduler=ExpRK5_simp).to("cuda", local_rank)

    ExpRKEDM4_pipe = RungeKuttaPipeline(model=eps_model, scheduler=ExpRKEDM4).to("cuda", local_rank)
    ExpRKEDM5_pipe = RungeKuttaPipeline(model=eps_model, scheduler=ExpRKEDM5).to("cuda", local_rank)
    ExpRKEDM4_mid_pipe  = RungeKuttaPipeline(model=eps_model, scheduler=ExpRKEDM4_mid).to("cuda", local_rank)
    ExpRKEDM4_simp_pipe = RungeKuttaPipeline(model=eps_model, scheduler=ExpRKEDM4_simp).to("cuda", local_rank)
    ExpRKEDM5_mid_pipe  = RungeKuttaPipeline(model=eps_model, scheduler=ExpRKEDM5_mid).to("cuda", local_rank)
    ExpRKEDM5_simp_pipe = RungeKuttaPipeline(model=eps_model, scheduler=ExpRKEDM5_simp).to("cuda", local_rank)

    # Load the dataset
    batch_size = args.bsize * dist.get_world_size()
    dataloader = None

    if args.dataset == "CIFAR10":
        image_shape = (3, 32, 32)
    elif args.dataset == "LSUN-bedroom":
        image_shape = (3, 256, 256)

    if local_rank == 0:
        if args.dataset == "CIFAR10":
            preprocess = torchvision.transforms.Compose([
                # Doesn't need resizing
                torchvision.transforms.ToTensor(),  # Convert to tensor (0, 1)
            ])
            dataset = torchvision.datasets.CIFAR10(root=args.datapath, train=True, download=True, transform=preprocess)
            #dataset = torch.utils.data.Subset(dataset, range(1*batch_size))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        elif args.dataset == "LSUN-bedroom":
            preprocess = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256,256)),
                torchvision.transforms.ToTensor(),  # Convert to tensor (0, 1)
            ])
            dataset = torchvision.datasets.ImageFolder(root=args.datapath, transform=preprocess)
            #subset = torch.utils.data.Subset(dataset, range(1*batch_size))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prepare to write on a CSV file
    fieldnames = ["timestamp", "model", "name", "nfe", "inference-time", "peak-gpu-memory", "fid"]
    os.makedirs(args.csvdir, exist_ok=True)
    csv_path = os.path.join(args.csvdir, f"evaluation-{args.dataset}.csv")
    write_header = not os.path.exists(csv_path)

    # Pipeline configurations
    pipes_dict = {
        "DDPM": DDPM_pipe,
        "DDIM": DDIM_pipe,

        "DPMSolver": DPMSolver_pipe,
        "DPMEDM": DPMEDM_pipe,
        "DPMOSch": DPMOSch_pipe,
        "DPMEDMOSch": DPMEDMOSch_pipe, 
        "DPMEDMOSch_high": DPMEDMOSch_high_pipe, 

        "RK1": rk1_pipe,
        "RK2": rk2_pipe,
        "RK3": rk3_pipe,
        "RK4": rk4_pipe,

        "RKEDM1": RKEDM1_pipe,
        "RKEDM2": RKEDM2_pipe,
        "RKEDM3": RKEDM3_pipe,
        "RKEDM": RKEDM_pipe,
        "RKfEDM1": RKfEDM1_pipe,
        "RKfEDM2": RKfEDM2_pipe,
        "RKfEDM3": RKfEDM3_pipe,
        "RKfEDM": RKfEDM_pipe,

        "RKOSch": RKOSch_pipe,
        "RKEDMOSch": RKEDMOSch_pipe,
        "RKEDMOSch_high": RKEDMOSch_high_pipe,

        "RKfEDMcomp": RKfEDMcomp_pipe,
        "RKfEDMcomp_high": RKfEDMcomp_high_pipe,

        "WiSPRK2": WiSPRK2_pipe,
        "WiSPRK3": WiSPRK3_pipe,
        "WiSPRK4": WiSPRK4_pipe,
        "WiSPRKEDM2": WiSPRKEDM2_pipe,
        "WiSPRKEDM3": WiSPRKEDM3_pipe,
        "WiSPRKEDM4": WiSPRKEDM4_pipe,

        "ExpRK4": ExpRK4_pipe,
        "ExpRK5": ExpRK5_pipe,
        "ExpRK4_mid" : ExpRK4_mid_pipe,
        "ExpRK4_simp": ExpRK4_simp_pipe,
        "ExpRK5_mid" : ExpRK5_mid_pipe,
        "ExpRK5_simp": ExpRK5_simp_pipe,
        
        "ExpRKEDM4": ExpRKEDM4_pipe,
        "ExpRKEDM5": ExpRKEDM5_pipe,
        "ExpRKEDM4_mid" : ExpRKEDM4_mid_pipe,
        "ExpRKEDM4_simp": ExpRKEDM4_simp_pipe,
        "ExpRKEDM5_mid" : ExpRKEDM5_mid_pipe,
        "ExpRKEDM5_simp": ExpRKEDM5_simp_pipe
    }

    # Run pipelines
    if local_rank == 0:
        print("Running pipelines")
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if local_rank == 0 and write_header:
            writer.writeheader()
            csvfile.flush()

        for solver in args.solvers:
            try:
                name, pipe = solver, pipes_dict[solver]
            except KeyError:
                print(f"Invalid solver: {solver}. Skipping to the next one")
                continue

            for nfe in args.nfes:
                if local_rank == 0:
                    print(f"{name} targeting {nfe} NFE:")

                steps = nfe_to_steps(name, nfe, pipe.scheduler)
                actual_nfe = steps_to_nfe(name, steps, pipe.scheduler)  # to account for rounding errors

                stats = calculate_stats(dataloader, image_shape, pipe, steps, local_rank)

                if local_rank == 0:
                    stats["timestamp"] = script_start_time
                    stats["model"] = args.model
                    stats["name"] = name
                    stats["nfe"] = actual_nfe  # display the true NFE

                    print(f"Target NFE      : {nfe}")
                    print(f"Rounded steps   : {steps}")
                    print(f"Actual NFE used : {actual_nfe}")
                    print(f"Inference time  : {stats['inference-time']} s")
                    print(f"Peak GPU memory : {stats['peak-gpu-memory']} MB")
                    print(f"FID Score       : {stats['fid']}")

                    writer.writerow(stats)
                    csvfile.flush()

                dist.barrier()

    dist.barrier()
    dist.destroy_process_group()
