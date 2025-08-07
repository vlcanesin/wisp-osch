import torch
from diffusers import UNet2DModel
from tqdm import tqdm

class CustomDiffusionPipeline():
    """Diffusion pipeline that takes a unet (UNet2DModel) and a scheduler as input"""
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

    def __call__(self, shape, num_inference_steps=1000, generator=None):
        image = torch.randn(shape, device=self.execution_device, generator=generator)
        self.scheduler.set_timesteps(num_inference_steps, device=self.execution_device)

        with torch.no_grad():
            for t in tqdm(self.scheduler.timesteps, desc="Denoising"):
                model_input = self.scheduler.scale_model_input(image, t)
                noise_pred = self.unet(model_input,t).sample
                image = self.scheduler.step(noise_pred, t, model_input, generator=generator).prev_sample

        return image

class RungeKuttaPipeline(CustomDiffusionPipeline):
    """Slightly modified pipeline class that handles the step function correctly"""
    def __call__(self, shape, num_inference_steps=1000, generator=None):
        image = torch.randn(shape, device=self.execution_device, generator=generator)
        self.scheduler.set_timesteps(num_inference_steps, device=self.execution_device)

        with torch.no_grad():
            for t in tqdm(self.scheduler.timesteps, desc="Denoising"):
                image = self.scheduler.step(t, image, self.unet).prev_sample

        return image