from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
import torch

class DPMRungeKuttaScheduler(SchedulerMixin):
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.sig_sq = torch.linspace(beta_start, beta_end, num_train_timesteps)  # linear noise schedule
        self.alphas = torch.sqrt(1.0 - self.sig_sq)  # DDPM-like

        log_alphas = torch.log(self.alphas)
        self.log_alphas_derivative = torch.zeros_like(log_alphas)
        self.log_alphas_derivative[:-1] = log_alphas[1:] - log_alphas[:-1]  # delta_t = 1
        self.log_alphas_derivative[-1] = self.log_alphas_derivative[-2]

        sig_sq_derivative = torch.zeros_like(self.sig_sq)
        sig_sq_derivative[:-1] = self.sig_sq[1:] - self.sig_sq[:-1]
        sig_sq_derivative[-1] = sig_sq_derivative[-2]
        self.eps_factor = (sig_sq_derivative - 2*self.log_alphas_derivative*self.sig_sq) / (2 * torch.sqrt(self.sig_sq))

    def set_timesteps(self, num_inference_steps, device="cpu"):
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps).long().to(device)

    def step(self, t, sample, model):
        """
        Runge-Kutta 4 integration step for reverse ODE
        """
        delta = self.get_delta_t(t)

        # ODE based on the DPMSolver paper
        def f(sample, t):
            t = t.round().long()
            eps = model(sample, t).sample
            return self.log_alphas_derivative[t] * sample + self.eps_factor[t] * eps

        k1 = f(sample, t)
        k2 = f(sample + 0.5 * delta * k1, t + 0.5 * delta)
        k3 = f(sample + 0.5 * delta * k2, t + 0.5 * delta)
        k4 = f(sample + delta * k3, t + delta)

        weighted_sum = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        prev_sample = sample + delta * weighted_sum
        return SchedulerOutput(prev_sample=prev_sample)

    def get_delta_t(self, t):
        t_index = (self.timesteps == t).nonzero().item()
        if t_index + 1 >= len(self.timesteps):
            return 0.0
        return self.timesteps[t_index + 1] - t  # always negative
