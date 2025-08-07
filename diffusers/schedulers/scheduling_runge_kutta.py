from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
import torch
import bisect

class RungeKuttaScheduler(SchedulerMixin):
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 sigma_min=0.002, sigma_max=80, rho=7, timestep_schedule="linear", order=4, use_order_scheduling=False):
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
        if use_order_scheduling and timestep_schedule != "edm":
            raise ValueError(f"Order scheduling implemented only for edm timestep schedule")
        self.use_order_scheduling = use_order_scheduling
        self.order_list = [4, 3, 2, 1]
        self.order_separators = [0.0, 0.326, 0.397, 0.493]

    def set_timesteps(self, num_inference_steps, device="cpu"):
        self.num_inference_steps = num_inference_steps
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

    def _get_timestep_index(self, t):
        eps = 0.0001
        return ((t-eps < self.timesteps) & (self.timesteps < t+eps)).nonzero().item()

    def _get_order_from_schedule(self, timestep):
        index = self._get_timestep_index(timestep)
        progress = index / self.num_inference_steps
        order_idx = bisect.bisect_right(self.order_separators, progress) - 1
        return self.order_list[order_idx]

    def _get_delta_t(self, t):
        t_index = self._get_timestep_index(t)
        if t_index + 1 >= len(self.timesteps):
            return 0.0
        return self.timesteps[t_index + 1] - t  # always negative

    def step(self, t, sample, model):
        """
        Runge-Kutta 4 integration step for reverse ODE
        """
        delta = self._get_delta_t(t)
        if self.use_order_scheduling:
            order = self._get_order_from_schedule(t)
        else:
            order = self.order

        weighted_sum = 0
        if order == 1:  # Euler
            k1 = self.f(sample, t, model)
            weighted_sum = k1
        elif order == 2:  # Heun
            k1 = self.f(sample, t, model)
            k2 = self.f(sample + 1 * delta * k1, t + 1 * delta, model)
            weighted_sum = (k1 + k2) / 2.0
        elif order == 3:  # RK3
            k1 = self.f(sample, t, model)
            k2 = self.f(sample + 0.5 * delta * k1, t + 0.5 * delta, model)
            k3 = self.f(sample + delta * (-1 * k1 + 2 * k2), t + 1 * delta, model)
            weighted_sum = (k1 + 4 * k2 + k3) / 6.0
        elif order == 4:  # RK4
            k1 = self.f(sample, t, model)
            k2 = self.f(sample + 0.5 * delta * k1, t + 0.5 * delta, model)
            k3 = self.f(sample + 0.5 * delta * k2, t + 0.5 * delta, model)
            k4 = self.f(sample + delta * k3, t + delta, model)
            weighted_sum = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        prev_sample = sample + delta * weighted_sum
        return SchedulerOutput(prev_sample=prev_sample)
