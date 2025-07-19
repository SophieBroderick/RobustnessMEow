import torch
import torch.nn as nn

from .nets import MLP
from .transforms import Preprocessing
from .distributions import ConditionalDiagLinearGaussian
from .flows import MaskedCondAffineFlow, CondScaling

def initializeFlow(log_sigma_max, log_sigma_min, action_sizes, state_sizes):
    dropout_rate_flow = 0.1
    dropout_rate_scale = 0.0
    layer_norm_flow = True
    layer_norm_scale = False
    hidden_layers = 2
    flow_layers = 2
    hidden_sizes = 64
    scale_hidden_sizes = 256

    # Construct the prior distribution and the linear transformation
    prior_list = [state_sizes] + [hidden_sizes]*hidden_layers + [action_sizes]
    log_scale = MLP(prior_list, init='zero')
    q0 = ConditionalDiagLinearGaussian(action_sizes, log_scale, LOG_SIGMA_MIN=log_sigma_min, LOG_SIGMA_MAX=log_sigma_max)

    # Construct normalizing flow
    flows = []
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(action_sizes)])
    for i in range(flow_layers):
        layers_list = [action_sizes+state_sizes] + [hidden_sizes]*hidden_layers + [action_sizes]
        s = None
        t1 = MLP(layers_list, init='orthogonal', dropout_rate=dropout_rate_flow, layernorm=layer_norm_flow)
        t2 = MLP(layers_list, init='orthogonal', dropout_rate=dropout_rate_flow, layernorm=layer_norm_flow)
        flows += [MaskedCondAffineFlow(b, t1, s)]
        flows += [MaskedCondAffineFlow(1 - b, t2, s)]

    # Construct the reward shifting function
    scale_list = [state_sizes] + [scale_hidden_sizes]*hidden_layers + [1]
    learnable_scale_1 = MLP(scale_list, init='zero', dropout_rate=dropout_rate_scale, layernorm=layer_norm_scale)
    learnable_scale_2 = MLP(scale_list, init='zero', dropout_rate=dropout_rate_scale, layernorm=layer_norm_scale)
    flows += [CondScaling(learnable_scale_1, learnable_scale_2)]

    # Construct the preprocessing layer
    flows += [Preprocessing()]
    return flows, q0

class FlowPolicy(nn.Module):
    def __init__(self, alpha, log_sigma_max, log_sigma_min, action_sizes, state_sizes, device):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.action_shape = action_sizes
        flows, q0 = initializeFlow(log_sigma_max, log_sigma_min, action_sizes, state_sizes)
        self.flows = nn.ModuleList(flows).to(self.device)
        self.prior = q0.to(self.device)

    def forward(self, obs, act):
        log_q = torch.zeros(act.shape[0], dtype=act.dtype, device=act.device)
        z = act
        for flow in self.flows:
            z, log_det = flow.forward(z, context=obs)
            log_q -= log_det
        return z, log_q

    def inverse(self, obs, act):
        log_q = torch.zeros(act.shape[0], dtype=act.dtype, device=act.device)
        z = act
        for flow in reversed(self.flows):
            z, log_det = flow.inverse(z, context=obs)
            log_q += log_det
        return z, log_q

    def sample(self, num_samples, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if deterministic:
            eps = torch.randn((num_samples,) + self.prior.shape, dtype=obs.dtype, device=obs.device)
            act, _ = self.prior.get_mean_std(eps, context=obs)
            log_q = self.prior.log_prob(act, context=obs)
        else:
            act, log_q = self.prior.sample(context=obs, num_samples=num_samples)
        a, log_det = self.forward(obs=obs, act=act)
        log_q -= log_det
        return a, log_q

    def log_prob(self, obs, act):
        z, log_q = self.inverse(obs=obs, act=act)
        log_q += self.prior.log_prob(z, context=obs)
        return log_q

    def get_qv(self, obs, act):
        q = torch.zeros((act.shape[0]), device=act.device)
        v = torch.zeros((act.shape[0]), device=act.device)
        z = act
        for flow in reversed(self.flows):
            z, q_, v_ = flow.get_qv(z, context=obs)
            q += q_
            v += v_
        q_, v_ = self.prior.get_qv(z, context=obs)
        q += q_
        v += v_
        q = q * self.alpha
        v = v * self.alpha
        return q[:, None], v[:, None]

    def get_v(self, obs):
        act = torch.zeros((obs.shape[0], self.action_shape), device=self.device)
        v = torch.zeros((act.shape[0]), device=act.device)
        z = act
        for flow in reversed(self.flows):
            z, _, v_ = flow.get_qv(z, context=obs)
            v += v_
        _, v_ = self.prior.get_qv(z, context=obs)
        v += v_
        v = v * self.alpha
        return v[:, None]