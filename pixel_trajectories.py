import argparse
import os, csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from matplotlib.ticker import MaxNLocator

from diffusers import (
    UNet2DModel, 
    RungeKuttaScheduler,
    DPMSolverSinglestepScheduler,
    ExpRungeKuttaScheduler
)

from models.SimpleUNet import SimpleUNet
from models.DiT import DiT_models

# Supported solvers per schedule
SCHEDULE_SUPPORT = {
    "linear": ["DPM-Solver-1", "DPM-Solver-2", "DPM-Solver-3", "RK1", "RK2", "RK3", "RK4", "expRK4s6"],
    "EDM":    ["DPM-Solver-1", "DPM-Solver-2", "DPM-Solver-3", "RK1", "RK2", "RK3", "RK4", "expRK4s6"],
    "tEDM":   ["RK1", "RK2", "RK3", "RK4", "expRK4s6"],  # only RK solvers
}

# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models",
        nargs="+",
        default=["unet_simple", "unet_diffusers", "dit_cifar"],
        choices=["unet_simple", "unet_diffusers", "dit_cifar"])

    parser.add_argument("--path", type=str, default="default")

    parser.add_argument("--solvers",
        nargs="+",
        default=["DPM-Solver-1", "DPM-Solver-2", "DPM-Solver-3", "RK1", "RK2", "RK3", "RK4", "expRK4s6"],
        choices=["DPM-Solver-1", "DPM-Solver-2", "DPM-Solver-3", "RK1", "RK2", "RK3", "RK4", "expRK4s6"])

    parser.add_argument("--schedules",
        nargs="+",
        default=["linear", "EDM", "tEDM"],
        choices=["linear", "EDM", "tEDM"])

    parser.add_argument("--metric",
        choices=["avg", "max"],
        default="avg")

    parser.add_argument("--steps", type=int, default=100)  # 1000
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="plots")

    return parser.parse_args()


# ============================================================
# MODEL FACTORY
# ============================================================

def load_model(name, args, device):
    path = args.path
    if name == "unet_simple":
        model = SimpleUNet()
        if path == "default":
            path = "DiffusionPretrained/checkpoints/diffusion_model_final.pth"
        ckpt = torch.load(
            path,
            map_location="cpu"
        )
        model.load_state_dict(ckpt["model_state_dict"])

    elif name == "unet_diffusers":
        if path == "default":
            path = "../07-Better-CIFAR10/pretrained/ddpm_ema_cifar10"
        model = UNet2DModel.from_pretrained(path, subfolder="unet")

    elif name == "dit_cifar":
        model = DiT_models['DiT-S/4'](
            input_size=args.image_size,
            num_classes=0  # unconditional generation
        )
        if path == "default":
            path = "DiT-CIFAR10/results/001-DiT-S-4/checkpoints/0280000.pt"
        assert os.path.isfile(path), f'Could not find DiT checkpoint at {path}'

        # Allow argparse.Namespace to be unpickled safely
        torch.serialization.add_safe_globals([argparse.Namespace])
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        model.load_state_dict(checkpoint)

    return model.eval().to(device)

# ============================================================
# METRIC
# ============================================================

def pixel_diff(x_next, x, metric):
    diff = torch.abs(x_next - x)
    if metric == "avg":
        return diff.mean(dim=(1,2,3))
    else:
        return diff.amax(dim=(1,2,3))


def summarize(diff_tensor):
    # diff_tensor: [B, T]
    mean = diff_tensor.mean(dim=0)
    std = diff_tensor.std(dim=0, unbiased=True)
    ci99 = 2.576 * std
    return mean, ci99

def summarize_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    mean = torch.tensor(df['mean_diff'].values)
    std = torch.tensor(df['std_diff'].values)
    ci99 = 2.576 * std
    return mean, ci99

def get_scheduler(solver_name, schedule, args, device):

    if solver_name == "DPM-Solver-1":
        scheduler = DPMSolverSinglestepScheduler(
            num_train_timesteps=1000, prediction_type="epsilon",
            solver_order=1, algorithm_type="dpmsolver", final_sigmas_type="sigma_min",
            use_karras_sigmas = (schedule=="EDM"),
            lower_order_final = True  # has a small influence in high NFE
        )
        scheduler.set_timesteps(num_inference_steps=args.steps, device=device)

    elif solver_name == "DPM-Solver-2":
        scheduler = DPMSolverSinglestepScheduler(
            num_train_timesteps=1000, prediction_type="epsilon",
            solver_order=2, algorithm_type="dpmsolver", final_sigmas_type="sigma_min",
            use_karras_sigmas = (schedule=="EDM"),
            lower_order_final = True  # has a small influence in high NFE
        )
        scheduler.set_timesteps(num_inference_steps=args.steps*2, device=device)
        ## NOTE: using args.steps*2 compensates for the fact that num_inference_steps
        # is actually the number of NFE used in the denoising process

    elif solver_name == "DPM-Solver-3":
        scheduler = DPMSolverSinglestepScheduler(
            num_train_timesteps=1000, prediction_type="epsilon",
            solver_order=3, algorithm_type="dpmsolver", final_sigmas_type="sigma_min",
            use_karras_sigmas = (schedule=="EDM"),
            lower_order_final=True  # has a small influence in high NFE
        )
        scheduler.set_timesteps(num_inference_steps=args.steps*3, device=device)
        ## NOTE: using args.steps*3 compensates for the fact that num_inference_steps
        # is actually the number of NFE used in the denoising process

    elif solver_name == "RK1":
        scheduler = RungeKuttaScheduler(
            num_train_timesteps=1000, timestep_schedule=schedule, order=1
        )
        scheduler.set_timesteps(num_inference_steps=args.steps, device=device)

    elif solver_name == "RK2":
        scheduler = RungeKuttaScheduler(
            num_train_timesteps=1000, timestep_schedule=schedule, order=2
        )
        scheduler.set_timesteps(num_inference_steps=args.steps, device=device)

    elif solver_name == "RK3":
        scheduler = RungeKuttaScheduler(
            num_train_timesteps=1000, timestep_schedule=schedule, order=3
        )
        scheduler.set_timesteps(num_inference_steps=args.steps, device=device)

    elif solver_name == "RK4":
        scheduler = RungeKuttaScheduler(
            num_train_timesteps=1000, timestep_schedule=schedule, order=4
        )
        scheduler.set_timesteps(num_inference_steps=args.steps, device=device)

    elif solver_name == "expRK4s6":
        scheduler = ExpRungeKuttaScheduler(
            num_train_timesteps=1000, timestep_schedule=schedule, order=4,
            quadrature="simpson"
        )
        scheduler.set_timesteps(num_inference_steps=args.steps, device=device)

    else:
        raise ValueError(solver_name)
    
    return scheduler


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run(model_name, solver_name, schedule, args, device):
    torch.manual_seed(args.seed)

    if solver_name not in SCHEDULE_SUPPORT.get(schedule, []):
        print(f"Solver {solver_name} doesn't support schedule {schedule}, skipping run")
        return None, None

    # -------------------------------
    # CSV setup
    # -------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(
        args.outdir,
        f"pixel_traj_{model_name}_{solver_name}_{schedule}.csv"
    )

    # -------------------------------
    # Check if results already exist
    # -------------------------------
    if os.path.isfile(csv_path):
        print(f"Loading cached results from {csv_path}")
        return summarize_from_csv(csv_path)

    header = [
        "step",
        "model",
        "solver",
        "schedule",
        "metric",
        "mean_diff",
        "std_diff",
    ]

    B, C, H, W = args.batch_size, 3, args.image_size, args.image_size
    model = load_model(model_name, args, device)

    # -------------------------------
    # Denoising loop
    # -------------------------------

    all_diffs = []
    
    for batch_idx in range(args.num_batches):
        print(f"Processing batch {batch_idx+1}/{args.num_batches}")
        
        scheduler = get_scheduler(solver_name, schedule, args, device)
        
        sample = torch.randn((B, C, H, W), device=device)
        last_step_sample = sample.clone().detach()
        batch_diffs = []

        with torch.no_grad():

            for i, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising")):

                if solver_name in ["RK1", "RK2", "RK3", "RK4", "expRK4s6"]:
                    # RK solvers: scheduler calls the model internally
                    next_sample = scheduler.step(t, sample, model).prev_sample

                else:
                    # ---------- model interfaces ----------
                    if model_name == "unet_diffusers":
                        noise_pred = model(sample, t).sample

                    elif model_name == "unet_simple":
                        t_batch = t.expand(sample.shape[0])
                        noise_pred = model(sample, t_batch)

                    elif model_name == "dit_cifar":
                        # DiT CIFAR10 (no guidance, no latent)
                        # assumes forward(x, t) -> epsilon
                        t_batch = t.expand(sample.shape[0])
                        noise_pred = model(sample, t_batch, None)

                        # DiT outputs [epsilon, variance]
                        C = sample.shape[1]
                        noise_pred = noise_pred[:, :C]  # keep epsilon only
    
                    else:
                        raise ValueError(model_name)

                    # ---------- scheduler step ----------
                    next_sample = scheduler.step(
                        noise_pred, t, sample
                    ).prev_sample

                # ---------- pixel diffs ----------
                step_diff = None

                if solver_name == "DPM-Solver-2":
                    if i % 2 == 1:
                        if last_step_sample is not None:
                            step_diff = pixel_diff(next_sample, last_step_sample, args.metric)
                            batch_diffs.append(step_diff)

                        # update cache ONLY at completed steps
                        last_step_sample = next_sample.clone()

                elif solver_name == "DPM-Solver-3":
                    if i % 3 == 2:
                        if last_step_sample is not None:
                            step_diff = pixel_diff(next_sample, last_step_sample, args.metric)
                            batch_diffs.append(step_diff)

                        # update cache ONLY at completed steps
                        last_step_sample = next_sample.clone()

                else:
                    step_diff = pixel_diff(next_sample, sample, args.metric)
                    batch_diffs.append(step_diff)

                sample = next_sample

        # Stack this batch's diffs: [B, T]
        all_diffs.append(torch.stack(batch_diffs, dim=1))

    # Concatenate all batches along batch dimension: [B*num_batches, T]
    all_diffs = torch.cat(all_diffs, dim=0)

    mean, ci = summarize(all_diffs)
    std = all_diffs.std(dim=0, unbiased=True)

    # -------------- write to csv --------------
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for step in range(mean.shape[0]):
            writer.writerow([
                step,
                model_name,
                solver_name,
                schedule,
                args.metric,
                mean[step].item(),
                std[step].item(),
            ])

    return mean, ci


# ============================================================
# PLOTTING
# ============================================================

def log_to_csv(csv_path, row, header=None):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if (not file_exists) and header is not None:
            writer.writerow(header)
        writer.writerow(row)

def plot_grid(results, args):
    os.makedirs(args.outdir, exist_ok=True)

    fig, axes = plt.subplots(
        len(args.schedules),
        len(args.solvers),
        figsize=(4 * len(args.solvers), 3 * len(args.schedules)),
        sharex=True,
        sharey=True,
    )

    if len(args.schedules) == 1:
        axes = np.expand_dims(axes, 0)
    if len(args.solvers) == 1:
        axes = np.expand_dims(axes, 1)  # add column axis

    for i, sched in enumerate(args.schedules):
        for j, solver in enumerate(args.solvers):
            ax = axes[i, j]

            # Check if solver supports this schedule
            if solver not in SCHEDULE_SUPPORT.get(sched, []):
                # Draw a placeholder
                ax.text(
                    0.5, 0.5,
                    "Not implemented",
                    ha="center", va="center",
                    fontsize=22,
                    color="gray",
                    transform=ax.transAxes
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor("#f0f0f0")  # optional: light gray background

                # Titles / labels
                if i == 0:
                    ax.set_title(solver, fontsize=26)
                if j == 0:
                    ax.set_ylabel(sched, fontsize=26)

                # Tick formatting
                ax.tick_params(axis="both", labelsize=22)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

                continue  # skip plotting

            # Normal plotting for supported solvers
            line_styles = ['-', '--', ':']  # solid, dashed, dotted

            for idx, (model_name, (mean, ci)) in enumerate(results[sched][solver].items()):
                x = np.arange(len(mean))
                ax.plot(
                    x,
                    mean.cpu(),
                    linewidth=2.2,
                    linestyle=line_styles[idx % len(line_styles)],  # cycle if >3 models
                    label=model_name,
                )
                ax.fill_between(
                    x,
                    (mean - ci).cpu(),
                    (mean + ci).cpu(),
                    alpha=0.2,
                )

            # Add vertical dashed line at peak
            peak_idx = torch.argmax(mean).item()  
            ax.axvline(
                peak_idx,
                color='gray',
                linestyle='--',
                linewidth=1.5,
                alpha=0.7
            )

            ax.text(
                int(peak_idx), ax.get_ylim()[0] - 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),  
                str(peak_idx + 1),  # 1-indexed
                ha='center',
                va='top',
                fontsize=16,
                color='gray'
            )

            # Titles / labels
            if i == 0:
                ax.set_title(solver, fontsize=26)
            if j == 0:
                ax.set_ylabel(sched, fontsize=26)

            # Tick formatting
            ax.tick_params(axis="both", labelsize=22)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(True, which="major", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "pixel_trajectories_grid.svg"))
    plt.close(fig)


def matplotlib_config():
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    matplotlib_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {s: {v: {} for v in args.solvers} for s in args.schedules}

    for sched in args.schedules:
        for solver in args.solvers:
            for model in args.models:
                print(f"[{sched} | {solver} | {model}]")
                mean, ci = run(model, solver, sched, args, device)
                results[sched][solver][model] = (mean, ci)

    plot_grid(results, args)


if __name__ == "__main__":
    main()
