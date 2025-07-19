import torch
import torch.nn as nn
import numpy as np
from normflows.distributions import BaseDistribution

class ConditionalDiagLinearGaussian(BaseDistribution):
    '''
    Conditional multivariate Gaussian distribution with diagonal covariance matrix
    '''
    def __init__(self, shape, log_scale=None, LOG_SIGMA_MIN=-5, LOG_SIGMA_MAX=-0.3):
        '''
        Args:
          shape: Tuple with shape of data, if int shape has one dimension
        '''
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.log_norm_factor = torch.tensor(-0.5 * np.prod(shape) * np.log(2 * np.pi))
        self.log_scale = log_scale
        self.LOG_SIGMA_MIN = LOG_SIGMA_MIN
        self.LOG_SIGMA_MAX = LOG_SIGMA_MAX
        self.reduce_dims = tuple(range(1, self.n_dim + 1))

    def get_mean_std(self, z, context):
        mean = torch.zeros_like(z)
        pre_logsigma = self.log_scale(context)
        scaled_logsigma = (torch.tanh(pre_logsigma) + 1) / 2.0
        logsigma = self.LOG_SIGMA_MIN + (self.LOG_SIGMA_MAX - self.LOG_SIGMA_MIN) * scaled_logsigma
        sigma = logsigma.exp()
        return mean, sigma

    def log_prob(self, z, context):
        mean, std = self.get_mean_std(z, context)
        log_p = torch.sum(torch.distributions.Normal(mean, std).log_prob(z), self.reduce_dims)
        return log_p

    def get_qv(self, z, context):
        mean, std = self.get_mean_std(z, context)
        q = -torch.sum(0.5 * torch.pow((z - mean) / std, 2), self.reduce_dims)
        v = self.log_norm_factor - torch.sum(torch.log(std), self.reduce_dims)
        return q, -v

    def sample(self, context, num_samples=1):
        eps = torch.randn((num_samples,) + self.shape, dtype=context.dtype, device=context.device)
        mean, std = self.get_mean_std(eps, context)
        z = mean + std * eps
        log_p = torch.sum(torch.distributions.Normal(mean, std).log_prob(z), self.reduce_dims)
        return z, log_p
