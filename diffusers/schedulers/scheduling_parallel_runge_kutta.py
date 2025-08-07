from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
import torch

class ParallelRungeKuttaScheduler(SchedulerMixin):
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 sigma_min=0.002, sigma_max=80, rho=7, timestep_schedule="linear", order=4):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.timestep_schedule = timestep_schedule
        if order not in [1, 2, 3, 4]:
            raise ValueError(f"Order {order} not supported")
        self.order = order

    def set_timesteps(self, num_inference_steps, device="cpu"):
        if self.timestep_schedule == "linear":
            self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps).to(device)
        elif self.timestep_schedule == "edm":
            timestep_indices = torch.arange(num_inference_steps, device=device)
            self.timesteps = (
                self.sigma_max ** (1 / self.rho) + 
                timestep_indices / (num_inference_steps - 1) * 
                (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
            self.timesteps *= self.num_train_timesteps / self.sigma_max  # normalization
        else:
            raise ValueError(f"Timestep schedule {self.timestep_schedule} not valid")

    def _interp1d(self, a, lb, ub, t):
        t = torch.clamp(t, lb, ub)
        tl = torch.floor(t).int()
        tu = torch.ceil(t).int()
        slope = a[tu] - a[tl]
        return a[tl] + slope * (t - tl) 

    # ODE based on the score-based models paper
    def f(self, sample, t, model):
        alpha_bar_t = self._interp1d(self.alpha_bars, 0, self.num_train_timesteps-1, t)
        beta_t = self._interp1d(self.betas, 0, self.num_train_timesteps - 1, t)
        sigma = torch.sqrt(1 - alpha_bar_t)
        eps = model(sample, t).sample
        return -0.5 * beta_t * (sample - eps / sigma)

    def step(self, t, sample, model):
        """
        Runge-Kutta 4 integration step for reverse ODE
        """
        delta = self.get_delta_t(t)

        weighted_sum = 0
        if self.order == 1:  # Euler
            k1 = self.f(sample, t, model)
            weighted_sum = k1
        elif self.order == 2:  # Approximate Heun
            k1 = self.f(sample, t, model)
            k2 = self.f(sample, t + 1 * delta, model)
            weighted_sum = (k1 + k2) / 2.0
        elif self.order == 3:  # Approximate RK3
            k1 = self.f(sample, t, model)
            k2 = self.f(sample, t + 0.5 * delta, model)
            k3 = self.f(sample, t + 1 * delta, model)
            weighted_sum = (k1 + 4 * k2 + k3) / 6.0
        elif self.order == 4:  # Approximate RK4 (3/8 rule)
            k1 = self.f(sample, t, model)
            k2 = self.f(sample, t + (1/3) * delta, model)
            k3 = self.f(sample, t + (2/3) * delta, model)
            k4 = self.f(sample, t + delta, model)
            weighted_sum = (k1 + 3 * k2 + 3 * k3 + k4) / 8.0

        prev_sample = sample + delta * weighted_sum
        return SchedulerOutput(prev_sample=prev_sample)

    def get_delta_t(self, t):
        eps = 0.0001
        t_index = ((t-eps < self.timesteps) & (self.timesteps < t+eps)).nonzero().item()
        if t_index + 1 >= len(self.timesteps):
            return 0.0
        return self.timesteps[t_index + 1] - t  # always negative
