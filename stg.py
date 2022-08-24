import torch
import torch.nn as nn
import math


class StochasticGates(nn.Module):
    def __init__(self, size, sigma, lam, gate_init=None):
        super().__init__()
        self.size = size
        if gate_init is None:
            mus = 1.0 * torch.ones(size)
        else:
            mus = torch.from_numpy(gate_init)
        self.mus = nn.Parameter(mus, requires_grad=True)
        self.sigma = sigma
        self.lam = lam
        self.sqrt_two = math.sqrt(2)

    def forward(self, x):
        gaussian = self.sigma * torch.randn(self.mus.size()) * self.training
        shifted_gaussian = self.mus + gaussian.to(x.device)
        z = self.make_bernoulli(shifted_gaussian)
        new_x = x * z
        return new_x

    @staticmethod
    def make_bernoulli(z):
        return torch.clamp(z, 0.0, 1.0)

    def get_reg(self):
        return self.lam * torch.sum((1 + torch.erf((self.mus / self.sigma) / self.sqrt_two)) / 2)

    def get_gates(self):
        return self.make_bernoulli(self.mus)
