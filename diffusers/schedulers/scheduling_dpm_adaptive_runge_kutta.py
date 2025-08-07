import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput

class DPMAdaptiveHeunScheduler(SchedulerMixin):
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 atol=0.0078, rtol=0.05, theta=0.9):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.rtol = rtol
        self.atol = atol
        self.theta = theta

    def set_timesteps(self, num_inference_steps, device="cpu"):
        # store discrete time points (if timestep size is not given)
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps).to(device)

    def f(self, sample, t, model):
        t = torch.tensor(t)
        t = torch.clamp(t.round().long(), 0, self.num_train_timesteps - 1)
        sigma = torch.sqrt(1 - self.alpha_bars[t])
        eps = model(sample, t).sample
        return -0.5 * self.betas[t] * (sample - eps / sigma)

    def norm_fn(self, v: torch.Tensor):
        # Compute L2 mean over spatial dimensions, like DPM
        flat = v.reshape((v.shape[0], -1))
        return torch.sqrt((flat ** 2).mean(dim=-1, keepdim=True))

    def step(self, x, t, h, model, x_prev):
        """
        Performs one adaptive Heun step (order = 2) with embedded Euler.
        Returns (x_new, new_h, x_lower) if step accepted; otherwise retries with smaller h.

        This follows the implementation of the adaptive DPM solver.
        """
        order = 2

        # Euler predictor (lower order)
        k1 = self.f(x, t, model)
        x_lower = x + h * k1  # first-order step

        # Heun corrector (higher order)
        k2 = self.f(x_lower, t + h, model)
        x_higher = x + 0.5 * h * (k1 + k2)

        # Compute delta (tolerance scaling)
        delta = torch.max(
            torch.ones_like(x) * self.atol,
            self.rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev))
        )

        # Error estimate
        E = self.norm_fn((x_higher - x_lower) / delta).max()
        new_h = self.theta * h * torch.pow(E, -1.0 / order)

        # print("E:", torch.max(E))

        if torch.all(E <= 1.0):
            # Accept step: update step size for next iteration
            return True, x_higher, new_h, x_lower
        else:
            # Reject step: shrink h and retry
            return False, x, new_h, x_prev