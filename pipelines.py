import torch
from diffusers import UNet2DModel
from tqdm import tqdm
import inspect

class CustomDiffusionPipeline():
    """Diffusion pipeline that takes a noise prediction model and a scheduler as input"""
    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.execution_device = torch.device("cpu")

    def to(self, device, local_rank):
        if device == "cuda":
            self.execution_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        else:
            self.execution_device = torch.device("cpu")

        self.model.to(self.execution_device)

        return self

    def _model_accepts_y(self):
        try:
            return "y" in inspect.signature(self.model.forward).parameters
        except (ValueError, TypeError):
            return False

    def _run_inference(self, sample, t):
        if hasattr(self.model, "sample"):  # maybe already a Scheduler-compatible wrapper
            eps = self.model(sample, t).sample
        elif hasattr(self.model, "__call__"):
            # Check if t is a scalar and needs batch expansion
            if len(t.shape) == 0 or t.shape == torch.Size([]):
                t_batch = t.expand(sample.shape[0])
            else:
                t_batch = t
            
            if self._model_accepts_y():
                out = self.model(sample, t_batch, None)
            else:
                out = self.model(sample, t_batch)
            
            # if diffusers UNet
            if hasattr(out, "sample"):
                eps = out.sample
            else:
                eps = out  # SimpleUNet or DiT: assume it returns prediction directly

                if eps.shape[1] == 2 * sample.shape[1]:  # Get only mean from DiT
                    eps = eps[:, : sample.shape[1]]

        return eps

    def __call__(self, shape, num_inference_steps=1000, generator=None):
        image = torch.randn(shape, device=self.execution_device, generator=generator)
        self.scheduler.set_timesteps(num_inference_steps, device=self.execution_device)

        with torch.no_grad():
            for t in tqdm(self.scheduler.timesteps, desc="Denoising"):
                model_input = self.scheduler.scale_model_input(image, t)
                noise_pred = self._run_inference(model_input, t)
                image = self.scheduler.step(noise_pred, t, model_input, generator=generator).prev_sample

        return image

class RungeKuttaPipeline(CustomDiffusionPipeline):
    """Slightly modified pipeline class that handles the step function correctly"""
    def __call__(self, shape, num_inference_steps=1000, generator=None):
        image = torch.randn(shape, device=self.execution_device, generator=generator)
        self.scheduler.set_timesteps(num_inference_steps, device=self.execution_device)

        with torch.no_grad():
            for t in tqdm(self.scheduler.timesteps, desc="Denoising"):
                image = self.scheduler.step(t, image, self.model).prev_sample

        return image
