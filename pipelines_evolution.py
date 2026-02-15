import torch
from diffusers import UNet2DModel
from tqdm import tqdm
import os
import uuid

def make_image_id(device_id: int, segment_idx: int):
    """
    Returns a unique string identifier for an image.
    Format: device{device}_seg{segment}_uuid{short_uuid}
    """
    short_uuid = uuid.uuid4().hex[:8]  # 8 hex chars
    return f"device{device_id}_seg{segment_idx}_uuid{short_uuid}"

class CustomDiffusionPipeline():
    """
    Diffusion pipeline that takes a UNet2DModel and a scheduler.
    Supports segmented trajectories (t_start -> t_end) with checkpointing.
    """
    def __init__(self, unet : UNet2DModel, scheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.execution_device = torch.device("cpu")

    def to(self, device, local_rank):
        if device == "cuda":
            self.execution_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        else:
            self.execution_device = torch.device("cpu")
        self.unet.to(self.execution_device)
        return self

    def __call__(self, shape, num_inference_steps=1000, generator=None,
                 start_image=None, segments=None, save_dir=None, local_rank=None):
        """
        segments: list of indices specifying where to segment the trajectory
                  e.g., [0, 50, 100, 200] generates segments 0->50, 50->100, 100->200
        start_image: optional initial image to start the trajectory
        save_dir: optional path to save checkpoint images after each segment
        local_rank: rank of the device (needed to generated the saving id)
        """
        if start_image is None:
            image = torch.randn(shape, device=self.execution_device, generator=generator)
        else:
            image = start_image.clone().to(self.execution_device)

        self.scheduler.set_timesteps(num_inference_steps, device=self.execution_device)
        timesteps = self.scheduler.timesteps

        return_intermediate = True
        if segments is None:
            segments = [0, len(timesteps)]
            return_intermediate = False

        os.makedirs(save_dir, exist_ok=True) if save_dir else None
        outputs = []

        # Iterate over segments
        for seg_idx in range(len(segments)-1):
            t_start_idx = segments[seg_idx]
            t_end_idx   = segments[seg_idx+1]
            timesteps_slice = timesteps[t_start_idx:t_end_idx]

            with torch.no_grad():
                for i, t in enumerate(tqdm(timesteps_slice, desc=f"Denoising segment {seg_idx+1}/{len(segments)-1}")):
                    model_input = self.scheduler.scale_model_input(image, t)
                    noise_pred = self.unet(model_input, t).sample
                    image = self.scheduler.step(noise_pred, t, model_input, generator=generator).prev_sample

            outputs.append(image.clone().detach().cpu())

            # Save checkpoint after segment
            if save_dir:
                image_id = make_image_id(local_rank, seg_idx)
                fname = os.path.join(save_dir, f"{image_id}.pt")
                torch.save(image.clone().detach().cpu(), fname)

        # Compatibility with evaluation.py
        if return_intermediate:
            return outputs, image  # outputs per segment, final image
        else:
            return outputs[0]

class RungeKuttaPipeline(CustomDiffusionPipeline):
    """
    Modified pipeline for RK solvers that require model in step call.
    """
    def __call__(self, shape, num_inference_steps=1000, generator=None,
                 start_image=None, segments=None, save_dir=None, local_rank=None):
        if start_image is None:
            image = torch.randn(shape, device=self.execution_device, generator=generator)
        else:
            image = start_image.clone().to(self.execution_device)

        self.scheduler.set_timesteps(num_inference_steps, device=self.execution_device)
        timesteps = self.scheduler.timesteps

        return_intermediate = True
        if segments is None:
            return_intermediate = False
            segments = [0, len(timesteps)]

        os.makedirs(save_dir, exist_ok=True) if save_dir else None
        outputs = []

        # Iterate over segments
        for seg_idx in range(len(segments)-1):
            t_start_idx = segments[seg_idx]
            t_end_idx   = segments[seg_idx+1]
            timesteps_slice = timesteps[t_start_idx:t_end_idx]

            with torch.no_grad():
                for t in tqdm(timesteps_slice, desc=f"RK segment {seg_idx+1}/{len(segments)-1}"):
                    image = self.scheduler.step(t, image, self.unet).prev_sample

            outputs.append(image.clone().detach().cpu())

            if save_dir:
                image_id = make_image_id(local_rank, seg_idx)
                fname = os.path.join(save_dir, f"{image_id}.pt")
                torch.save(image.clone().detach().cpu(), fname)

        # Compatibility with evaluation.py
        if return_intermediate:
            return outputs, image  # outputs per segment, final image
        else:
            return outputs[0]
