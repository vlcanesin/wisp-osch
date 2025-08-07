import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput

class AdaptiveHeunScheduler(SchedulerMixin):
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 rtol=1e-3, atol=1e-5, safety=0.9, min_factor=0.2, max_factor=5.0):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.rtol = rtol
        self.atol = atol
        self.safety = safety
        self.min_factor = min_factor
        self.max_factor = max_factor

    def _interp1d(self, a, lb, ub, t):
        t = torch.clamp(t, lb, ub)
        tl = torch.floor(t).int()
        tu = torch.ceil(t).int()
        slope = a[tu] - a[tl]
        return a[tl] + slope * (t - tl) 

    def f(self, sample, t, model):
        alpha_bar_t = self._interp1d(self.alpha_bars, 0, self.num_train_timesteps-1, t)
        beta_t = self._interp1d(self.betas, 0, self.num_train_timesteps - 1, t)
        sigma = torch.sqrt(1 - alpha_bar_t)
        eps = model(sample, t).sample
        return -0.5 * beta_t * (sample - eps / sigma)

    def step(self, t, sample, model, h, nfe_used):
        """
        One adaptive Heun (improved Euler) step with embedded Euler for error estimation.
        Returns (SchedulerOutput(prev_sample), new_h)
        """

        # slope at beginning
        k1 = self.f(sample, t, model)
        nfe_used += 1
        # Euler predictor
        y_euler = sample + h * k1

        # slope at end
        k2 = self.f(y_euler, t + h, model)
        nfe_used += 1
        # Heun corrector (average slopes)
        y_heun = sample + 0.5 * h * (k1 + k2)

        # local error estimate
        err = torch.abs(y_heun - y_euler)
        tol = self.atol + self.rtol * torch.maximum(torch.abs(y_heun), torch.abs(y_euler))
        err_ratio = torch.max(err / tol)

        # compute step size factor
        if err_ratio == 0:
            factor = self.max_factor
        else:
            # order = 2 for Heun
            factor = self.safety * (1.0 / err_ratio) ** 0.5
            factor = max(self.min_factor, min(self.max_factor, factor))
        new_h = h * factor

        # decide to accept or reject
        if err_ratio <= 1.0:
            # accept step
            return SchedulerOutput(prev_sample=y_heun), new_h, nfe_used
        else:
            # reject step and retry with smaller h
            return self.step(t, sample, model, h=new_h, nfe_used=nfe_used)