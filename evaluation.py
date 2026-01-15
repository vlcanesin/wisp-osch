import os

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

# Local version of the diffusers library
from diffusers import (
    UNet2DModel, 
    RungeKuttaScheduler, 
    DDPMScheduler, 
    DDIMScheduler, 
    DPMSolverSinglestepScheduler, 
    ParallelRungeKuttaScheduler, 
    ExpRungeKuttaScheduler,
    DPMSolverComposedScheduler
)

import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance

from torch_timer import Timer
from pipelines import CustomDiffusionPipeline, RungeKuttaPipeline

import torchvision

import csv
from datetime import datetime

import argparse

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

if __name__ == "__main__":

    script_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description="Parameters used for controling the tests")

    parser.add_argument('--solvers', type=str, nargs='+', help="List of solvers", required=True)
    parser.add_argument('--steps', type=int, nargs='+', help="List of steps", required=True)
    parser.add_argument('--dataset', type=str, help="Name of the dataset used", required=True)
    parser.add_argument('--bsize', type=int, help="Batch size (per GPU)", required=True)
    parser.add_argument('--csvdir', type=str, help="CSV output directory", default="./csv")
    parser.add_argument('--modelpath', type=str, help="Cached model path", required=True)
    parser.add_argument('--datapath', type=str, help="Dataset path for FID calculation", required=True)

    args = parser.parse_args()

    assert args.dataset in ["CIFAR10", "LSUN-bedroom"], "Invalid dataset. Choose one of the following: CIFAR10 or LSUN-bedroom"

    local_rank = setup_distributed()

    # Load components
    if local_rank == 0:
        print("Loading components")

    if args.dataset == "CIFAR10":
        eps_unet = UNet2DModel.from_pretrained(args.modelpath, subfolder="unet")
    elif args.dataset == "LSUN-bedroom":
        eps_unet = UNet2DModel.from_pretrained("google/ddpm-bedroom-256", cache_dir=args.modelpath)
    eps_unet.eval()

    DDPM = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    DDIM = DDIMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    DPMSolver = DPMSolverSinglestepScheduler(
        num_train_timesteps=1000, prediction_type="epsilon",
        solver_order=3, algorithm_type="dpmsolver++", final_sigmas_type="sigma_min",
        lower_order_final=True
    )
    DPMComposed = DPMSolverComposedScheduler(
        num_train_timesteps=1000, prediction_type="epsilon",
        solver_order=3, algorithm_type="dpmsolver++", final_sigmas_type="sigma_min",
        lower_order_final=True
    )

    rk1 = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="linear", order=1, use_order_scheduling=False)
    rk2 = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="linear", order=2, use_order_scheduling=False)
    rk3 = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="linear", order=3, use_order_scheduling=False)
    rk4 = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="linear", order=4, use_order_scheduling=False)

    RKEDM = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="edm", order=4, use_order_scheduling=False)
    RKcomp = RungeKuttaScheduler(num_train_timesteps=1000, timestep_schedule="edm", use_order_scheduling=True)
    
    pRK2 = ParallelRungeKuttaScheduler(num_train_timesteps=1000, order=2)
    pRK3 = ParallelRungeKuttaScheduler(num_train_timesteps=1000, order=3)
    pRK4 = ParallelRungeKuttaScheduler(num_train_timesteps=1000, order=4)

    ExpRK4 = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=4)
    ExpRK5 = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=5)

    ExpRK4_mid  = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=4, quadrature="midpoint")
    ExpRK4_simp = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=4, quadrature="simpson")
    ExpRK5_mid  = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=5, quadrature="midpoint")
    ExpRK5_simp = ExpRungeKuttaScheduler(num_train_timesteps=1000, order=5, quadrature="simpson")

    # Create the pipelines
    if local_rank == 0:
        print("Creating pipelines")
    DDPM_pipe = CustomDiffusionPipeline(unet=eps_unet, scheduler=DDPM).to("cuda", local_rank)
    DDIM_pipe = CustomDiffusionPipeline(unet=eps_unet, scheduler=DDIM).to("cuda", local_rank)
    DPMSolver_pipe = CustomDiffusionPipeline(unet=eps_unet, scheduler=DPMSolver).to("cuda", local_rank)
    DPMComposed_pipe = CustomDiffusionPipeline(unet=eps_unet, scheduler=DPMComposed).to("cuda", local_rank)

    rk1_pipe = RungeKuttaPipeline(unet=eps_unet, scheduler=rk1).to("cuda", local_rank)
    rk2_pipe = RungeKuttaPipeline(unet=eps_unet, scheduler=rk2).to("cuda", local_rank)
    rk3_pipe = RungeKuttaPipeline(unet=eps_unet, scheduler=rk3).to("cuda", local_rank)
    rk4_pipe = RungeKuttaPipeline(unet=eps_unet, scheduler=rk4).to("cuda", local_rank)

    RKEDM_pipe = RungeKuttaPipeline(unet=eps_unet, scheduler=RKEDM).to("cuda", local_rank)
    RKcomp_pipe = RungeKuttaPipeline(unet=eps_unet, scheduler=RKcomp).to("cuda", local_rank)

    pRK2_pipe = RungeKuttaPipeline(unet=eps_unet, scheduler=pRK2).to("cuda", local_rank)
    pRK3_pipe = RungeKuttaPipeline(unet=eps_unet, scheduler=pRK3).to("cuda", local_rank)
    pRK4_pipe = RungeKuttaPipeline(unet=eps_unet, scheduler=pRK4).to("cuda", local_rank)

    ExpRK4_pipe = RungeKuttaPipeline(unet=eps_unet, scheduler=ExpRK4).to("cuda", local_rank)
    ExpRK5_pipe = RungeKuttaPipeline(unet=eps_unet, scheduler=ExpRK5).to("cuda", local_rank)

    ExpRK4_mid_pipe  = RungeKuttaPipeline(unet=eps_unet, scheduler=ExpRK4_mid).to("cuda", local_rank)
    ExpRK4_simp_pipe = RungeKuttaPipeline(unet=eps_unet, scheduler=ExpRK4_simp).to("cuda", local_rank)
    ExpRK5_mid_pipe  = RungeKuttaPipeline(unet=eps_unet, scheduler=ExpRK5_mid).to("cuda", local_rank)
    ExpRK5_simp_pipe = RungeKuttaPipeline(unet=eps_unet, scheduler=ExpRK5_simp).to("cuda", local_rank)

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
    fieldnames = ["timestamp", "name", "num-inference-steps", "inference-time", "peak-gpu-memory", "fid"]
    os.makedirs(args.csvdir, exist_ok=True)
    csv_path = os.path.join(args.csvdir, f"evaluation-{args.dataset}.csv")
    write_header = not os.path.exists(csv_path)

    # Pipeline configurations
    pipes_dict = {
        "DDPM": DDPM_pipe,
        "DDIM": DDIM_pipe,
        "DPMSolver": DPMSolver_pipe,
        "DPMComposed": DPMComposed_pipe,

        "RK1": rk1_pipe,
        "RK2": rk2_pipe,
        "RK3": rk3_pipe,
        "RK4": rk4_pipe,

        "RKEDM": RKEDM_pipe,
        "RKcomp": RKcomp_pipe,

        "pRK2": pRK2_pipe,
        "pRK3": pRK3_pipe,
        "pRK4": pRK4_pipe,

        "ExpRK4": ExpRK4_pipe,
        "ExpRK5": ExpRK5_pipe,

        "ExpRK4_mid" : ExpRK4_mid_pipe,
        "ExpRK4_simp": ExpRK4_simp_pipe,
        "ExpRK5_mid" : ExpRK5_mid_pipe,
        "ExpRK5_simp": ExpRK5_simp_pipe
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

            for step in args.steps:
                if local_rank == 0:
                    print(f"{name} with {step} steps:")
                
                stats = calculate_stats(dataloader, image_shape, pipe, step, local_rank)
                
                if local_rank == 0:
                    stats["timestamp"] = script_start_time
                    stats["name"] = name
                    stats["num-inference-steps"] = step
                    print(f"Inference time  : {stats["inference-time"]} s")
                    print(f"Peak GPU memory : {stats["peak-gpu-memory"]} MB")
                    print(f"FID Score       : {stats["fid"]}")

                    writer.writerow(stats)
                    csvfile.flush()

    dist.barrier()
    dist.destroy_process_group()
