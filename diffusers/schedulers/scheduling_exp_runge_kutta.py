from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
import torch

class ExpRungeKuttaScheduler(SchedulerMixin):
    """
    Implementation of the exponential methods from the paper:
    'Efficient exponential Runge-Kutta methods of high order: construction and implementation' 
    """
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, order=4, quadrature="linear"):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        if order not in [1, 4, 5]:
            raise ValueError(f"Order {order} not supported")
        self.order = order
        if quadrature not in ["linear", "midpoint", "simpson"]:
            raise ValueError(f"Quadrature {quadrature} not supported")
        self.quadrature = quadrature

        # Parameters taken from the paper
        self.rk4c2 = 1/2
        self.rk4c3 = 1/2
        self.rk4c4 = 1/3
        self.rk4c5 = 5/6
        self.rk4c6 = 1/3

        self.rk5c2 = 1/2
        self.rk5c3 = 1/2
        self.rk5c4 = 1/3
        self.rk5c5 = 1/2
        self.rk5c6 = 1/3
        self.rk5c7 = 1/4
        self.rk5c8 = 3/10
        self.rk5c9 = 3/4
        self.rk5c10 = 1

        # For the numerical stability of the phi functions
        self.EPS = 0.5

    def set_timesteps(self, num_inference_steps, device="cpu"):
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps).to(device)

    def _phi_1(self, z):
        direct = (torch.exp(z) - 1) / z
        series = 1 + z/2 + z**2/6 + z**3/24
        return torch.where(torch.abs(z) < self.EPS, series, direct)
    
    def _phi_2(self, z):
        direct = (torch.exp(z) - z - 1) / z**2
        series = 1/2 + z/6 + z**2/24 + z**3/120
        return torch.where(torch.abs(z) < self.EPS, series, direct)
    
    def _phi_3(self, z):
        direct = 1/2 * (2*torch.exp(z) - z**2 - 2*z - 2) / z**3
        series = 1/6 + z/24 + z**2/120 + z**3/720
        return torch.where(torch.abs(z) < self.EPS, series, direct)
    
    def _phi_4(self, z):
        direct = (1/6 * (6*torch.exp(z) - z**3 - 3*z**2 - 6*z - 6) / z**4)
        series = 1/24 + z/120 + z**2/720 + z**3/5040
        return torch.where(torch.abs(z) < self.EPS, series, direct)

    def _interp1d(self, a, lb, ub, t):
        t = torch.clamp(t, lb, ub)
        tl = torch.floor(t).int()
        tu = torch.ceil(t).int()
        slope = a[tu] - a[tl]
        return a[tl] + slope * (t - tl) 

    def _A(self, t):
        return -0.5 * self._interp1d(self.betas, 0, self.num_train_timesteps-1, t)
    
    def _g(self, t, sample, model):
        bt = self._interp1d(self.betas, 0, self.num_train_timesteps-1, t)
        abt = self._interp1d(self.alpha_bars, 0, self.num_train_timesteps-1, t)
        return bt * model(sample,t).sample / (2 * torch.sqrt(1 - abt))

    def _quad_A(self, t, h):
        if self.quadrature == "midpoint":
            return self._A(t+h/2)
        elif self.quadrature == "simpson":
            return 1/6 * (self._A(t) + 4*self._A(t+h/2) + self._A(t+h))
        else:
            return self._A(t)

    def _expRK1s1(self, t, sample, model):
        h = self.get_delta_t(t)
        A = self._quad_A(t, h)

        g0 = self._g(t, sample, model)
        F = A * sample + g0

        prev_sample = sample + self._phi_1(h*A)*h*F
        return prev_sample

    def _expRK4s6(self, t, sample, model):
        h = self.get_delta_t(t)
        A = self._quad_A(t, h)

        # expRK4s6
        g0 = self._g(t, sample, model)  # 1 NFE
        F = A * sample + g0

        Un2 = sample + self._phi_1(self.rk4c2*h*A)*self.rk4c2*h*F

        Dn2 = self._g(t + self.rk4c2*h, Un2, model) - g0  # 2 NFEs
        Un3 = sample + \
              self._phi_1(self.rk4c3*h*A)*self.rk4c3*h*F + \
              self._phi_2(self.rk4c3*h*A)*self.rk4c3**2/self.rk4c2*h*Dn2
        Un4 = sample + \
              self._phi_1(self.rk4c4*h*A)*self.rk4c4*h*F + \
              self._phi_2(self.rk4c4*h*A)*self.rk4c4**2/self.rk4c2*h*Dn2
        
        # Both of these can be computed in parallel,
        # accounting for 1 sequential NFE
        Dn3 = self._g(t + self.rk4c3*h, Un3, model) - g0  # 3 NFEs
        Dn4 = self._g(t + self.rk4c4*h, Un4, model) - g0  # 4 NFEs 
        Un5 = sample + \
              self._phi_1(self.rk4c5*h*A)*self.rk4c5*h*F + \
              self._phi_2(self.rk4c5*h*A)*self.rk4c5**2/(self.rk4c3-self.rk4c4)*h*(-self.rk4c4/self.rk4c3*Dn3 + self.rk4c3/self.rk4c4*Dn4) + \
              self._phi_3(self.rk4c5*h*A)*2*self.rk4c5**3/(self.rk4c3-self.rk4c4)*h*(1/self.rk4c3*Dn3 - 1/self.rk4c4*Dn4)
        Un6 = sample + \
              self._phi_1(self.rk4c6*h*A)*self.rk4c6*h*F + \
              self._phi_2(self.rk4c6*h*A)*self.rk4c6**2/(self.rk4c3-self.rk4c4)*h*(-self.rk4c4/self.rk4c3*Dn3 + self.rk4c3/self.rk4c4*Dn4) + \
              self._phi_3(self.rk4c6*h*A)*2*self.rk4c6**3/(self.rk4c3-self.rk4c4)*h*(1/self.rk4c3*Dn3 - 1/self.rk4c4*Dn4)
        
        # Both of these can be computed in parallel,
        # accounting for 1 sequential NFE
        Dn5 = self._g(t + self.rk4c5*h, Un5, model) - g0  # 5 NFEs
        Dn6 = self._g(t + self.rk4c6*h, Un6, model) - g0  # 6 NFEs
        prev_sample = sample + \
                      self._phi_1(h*A)*h*F + \
                      self._phi_2(h*A)*1/(self.rk4c5-self.rk4c6)*h*(-self.rk4c6/self.rk4c5*Dn5 + self.rk4c5/self.rk4c6*Dn6) + \
                      self._phi_3(h*A)*2/(self.rk4c5-self.rk4c6)*h*(1/self.rk4c5*Dn5 - 1/self.rk4c6*Dn6)

        return prev_sample

    def _expRK5s10(self, t, sample, model):
        h = self.get_delta_t(t)
        A = self._quad_A(t, h)

        # expRK5s10
        g0 = self._g(t, sample, model)  # 1 NFE
        F = A * sample + g0

        Un2 = sample + self._phi_1(self.rk5c2*h*A)*self.rk5c2*h*F

        Dn2 = self._g(t + self.rk5c2*h, Un2, model) - g0  # 2 NFEs
        Un3 = sample + \
              self._phi_1(self.rk5c3*h*A)*self.rk5c3*h*F + \
              self._phi_2(self.rk5c3*h*A)*self.rk5c3**2/self.rk5c2*h*Dn2
        Un4 = sample + \
              self._phi_1(self.rk5c4*h*A)*self.rk5c4*h*F + \
              self._phi_2(self.rk5c4*h*A)*self.rk5c4**2/self.rk5c2*h*Dn2
        
        # Both of these can be computed in parallel,
        # accounting for 1 sequential NFE
        Dn3 = self._g(t + self.rk5c3*h, Un3, model) - g0  # 3 NFEs
        Dn4 = self._g(t + self.rk5c4*h, Un4, model) - g0  # 4 NFEs 
        coeff1 = self.rk5c4/(self.rk5c3*(self.rk5c4-self.rk5c3))
        coeff2 = self.rk5c3/(self.rk5c4*(self.rk5c3-self.rk5c4))
        coeff3 = 2/(self.rk5c3*(self.rk5c3-self.rk5c4))
        coeff4 = 2/(self.rk5c4*(self.rk5c3-self.rk5c4))
        Un5 = sample + \
              self._phi_1(self.rk5c5*h*A)*self.rk5c5*h*F + \
              self._phi_2(self.rk5c5*h*A)*self.rk5c5**2*h*(coeff1*Dn3+coeff2*Dn4) + \
              self._phi_3(self.rk5c5*h*A)*self.rk5c5**3*h*(coeff3*Dn3-coeff4*Dn4)
        Un6 = sample + \
              self._phi_1(self.rk5c6*h*A)*self.rk5c6*h*F + \
              self._phi_2(self.rk5c6*h*A)*self.rk5c6**2*h*(coeff1*Dn3+coeff2*Dn4) + \
              self._phi_3(self.rk5c6*h*A)*self.rk5c6**3*h*(coeff3*Dn3-coeff4*Dn4)
        Un7 = sample + \
              self._phi_1(self.rk5c7*h*A)*self.rk5c7*h*F + \
              self._phi_2(self.rk5c7*h*A)*self.rk5c7**2*h*(coeff1*Dn3+coeff2*Dn4) + \
              self._phi_3(self.rk5c7*h*A)*self.rk5c7**3*h*(coeff3*Dn3-coeff4*Dn4)

        # All of these can be computed in parallel,
        # accounting for 1 sequential NFE
        Dn5 = self._g(t + self.rk5c5*h, Un5, model) - g0  # 5 NFEs
        Dn6 = self._g(t + self.rk5c6*h, Un6, model) - g0  # 6 NFEs 
        Dn7 = self._g(t + self.rk5c7*h, Un7, model) - g0  # 7 NFEs
        d5 = self.rk5c5*(self.rk5c5-self.rk5c6)*(self.rk5c5-self.rk5c7)
        d6 = self.rk5c6*(self.rk5c6-self.rk5c5)*(self.rk5c6-self.rk5c7)
        d7 = self.rk5c7*(self.rk5c7-self.rk5c5)*(self.rk5c7-self.rk5c6)
        a5 = self.rk5c6*self.rk5c7/d5
        a6 = self.rk5c5*self.rk5c7/d6
        a7 = self.rk5c5*self.rk5c6/d7
        b5 = 2*(self.rk5c6+self.rk5c7)/d5
        b6 = 2*(self.rk5c5+self.rk5c7)/d6
        b7 = 2*(self.rk5c5+self.rk5c6)/d7
        g5 = 6/d5
        g6 = 6/d6
        g7 = 6/d7
        Un8 = sample + \
              self._phi_1(self.rk5c8*h*A)*self.rk5c8*h*F + \
              self._phi_2(self.rk5c8*h*A)*self.rk5c8**2*h*(a5*Dn5+a6*Dn6+a7*Dn7) + \
              self._phi_3(self.rk5c8*h*A)*self.rk5c8**3*h*(b5*Dn5-b6*Dn6-b7*Dn7) + \
              self._phi_4(self.rk5c8*h*A)*self.rk5c8**4*h*(g5*Dn5+g6*Dn6+g7*Dn7)
        Un9 = sample + \
              self._phi_1(self.rk5c9*h*A)*self.rk5c9*h*F + \
              self._phi_2(self.rk5c9*h*A)*self.rk5c9**2*h*(a5*Dn5+a6*Dn6+a7*Dn7) + \
              self._phi_3(self.rk5c9*h*A)*self.rk5c9**3*h*(b5*Dn5-b6*Dn6-b7*Dn7) + \
              self._phi_4(self.rk5c9*h*A)*self.rk5c9**4*h*(g5*Dn5+g6*Dn6+g7*Dn7)
        Un10 = sample + \
              self._phi_1(self.rk5c10*h*A)*self.rk5c10*h*F + \
              self._phi_2(self.rk5c10*h*A)*self.rk5c10**2*h*(a5*Dn5+a6*Dn6+a7*Dn7) + \
              self._phi_3(self.rk5c10*h*A)*self.rk5c10**3*h*(b5*Dn5-b6*Dn6-b7*Dn7) + \
              self._phi_4(self.rk5c10*h*A)*self.rk5c10**4*h*(g5*Dn5+g6*Dn6+g7*Dn7)
        
        # All of these can be computed in parallel,
        # accounting for 1 sequential NFE
        Dn8 = self._g(t + self.rk5c8*h, Un8, model) - g0  # 8 NFEs
        Dn9 = self._g(t + self.rk5c9*h, Un9, model) - g0  # 9 NFEs 
        Dn10 = self._g(t + self.rk5c10*h, Un10, model) - g0  # 10 NFEs
        d8 = self.rk5c8*(self.rk5c8-self.rk5c9)*(self.rk5c8-self.rk5c10)
        d9 = self.rk5c9*(self.rk5c9-self.rk5c8)*(self.rk5c9-self.rk5c10)
        d10 = self.rk5c10*(self.rk5c10-self.rk5c8)*(self.rk5c10-self.rk5c9)
        a8 = self.rk5c9*self.rk5c10/d8
        a9 = self.rk5c8*self.rk5c10/d9
        a10 = self.rk5c8*self.rk5c9/d10
        b8 = 2*(self.rk5c9+self.rk5c10)/d8
        b9 = 2*(self.rk5c8+self.rk5c10)/d9
        b10 = 2*(self.rk5c8+self.rk5c9)/d10
        g8 = 6/d8
        g9 = 6/d9
        g10 = 6/d10
        prev_sample = sample + \
                      self._phi_1(h*A)*h*F + \
                      self._phi_2(h*A)*h*(a8*Dn8+a9*Dn9+a10*Dn10) - \
                      self._phi_3(h*A)*h*(b8*Dn8+b9*Dn9+b10*Dn10) + \
                      self._phi_4(h*A)*h*(g8*Dn8+g9*Dn9+g10*Dn10)
        
        return prev_sample

    def step(self, t, sample, model):
        if self.order == 1:
            return SchedulerOutput(prev_sample=self._expRK1s1(t, sample, model))
        elif self.order == 4:
            return SchedulerOutput(prev_sample=self._expRK4s6(t, sample, model))
        elif self.order == 5:
            return SchedulerOutput(prev_sample=self._expRK5s10(t, sample, model))

    def get_delta_t(self, t):
        eps = 0.0001
        t_index = ((t-eps < self.timesteps) & (self.timesteps < t+eps)).nonzero().item()
        if t_index + 1 >= len(self.timesteps):
            return 0.0
        return self.timesteps[t_index + 1] - t  # always negative
