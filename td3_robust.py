import os
import random
import time
import yaml
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import robust_gymnasium as robust_gym
from robust_gymnasium.configs.robust_setting import get_config
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Config:
    seed: int
    env_id: str
    total_timesteps: int
    learning_rate: float
    buffer_size: int
    gamma: float
    tau: float
    batch_size: int
    policy_noise: float
    exploration_noise: float
    learning_starts: int
    policy_frequency: int
    noise_clip: float
    description: str = ''
    noise_factor: str = None
    noise_type: str = None
    llm_disturb_interval: int = None

    @staticmethod
    def from_yaml(path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return Config(**data)

class RobustRescaleActionWrapper:
    def __init__(self, env, min_action=-1.0, max_action=1.0):
        self.env = env
        self.orig_low = env.action_space.low
        self.orig_high = env.action_space.high
        self.min_action = min_action
        self.max_action = max_action
        self.action_space = gym.spaces.Box(low=min_action, high=max_action, shape=self.orig_low.shape, dtype=np.float32)
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action_input):
        if isinstance(action_input, dict):
            raw_action = action_input["action"]
            config = action_input.get("robust_config", None)
        else:
            raw_action = action_input
            config = None

        scaled_action = self.orig_low + 0.5 * (raw_action + 1.0) * (self.orig_high - self.orig_low)
        input_dict = {"action": scaled_action}
        if config is not None:
            input_dict["robust_config"] = config

        return self.env.step(input_dict)

    def __getattr__(self, name):
        return getattr(self.env, name)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, act_dim)
        self.action_scale = 1.0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x)) * self.action_scale

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def evaluate(envs, actor, config, device):
    actor.eval()
    returns = np.zeros((envs.unwrapped.num_envs,))
    dones = np.zeros_like(returns, dtype=bool)
    obs, _ = envs.reset(seed=range(envs.unwrapped.num_envs))

    with torch.no_grad():
        while not all(dones):
            actions = actor(torch.Tensor(obs).to(device)).cpu().numpy()
            args = get_config().parse_args()
            #args.env_name = config.env_id
            #args.env_name = "Ant-v4"
            #args.noise_factor = config.noise_factor
            #args.noise_type = config.noise_type
            #args.llm_disturb_interval = config.llm_disturb_interval
            robust_input = [{"action": a, "robust_config": args} for a in actions]
            obs, rewards, terminated, truncated, _ = envs.step(robust_input)
            done = terminated | truncated
            returns += rewards * (~dones)
            dones |= done

    return returns.mean()

def train(config):
    run_name = f"{config.env_id}_td3_{config.seed}_{config.noise_factor}_{int(time.time())}"
    config.description = f"runs/{run_name}"
    writer = SummaryWriter(config.description)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_env():
        env = robust_gym.make(config.env_id)
        env = RobustRescaleActionWrapper(env)

        env.action_space = gym.spaces.Box(
            low=np.array(env.action_space.low, dtype=np.float32),
            high=np.array(env.action_space.high, dtype=np.float32),
            dtype=np.float32,
        )
        env.observation_space = gym.spaces.Box(
            low=np.array(env.observation_space.low, dtype=np.float32),
            high=np.array(env.observation_space.high, dtype=np.float32),
            dtype=np.float32,
        )

        return env

    envs = gym.vector.SyncVectorEnv([lambda: make_env()])
    test_envs = gym.vector.SyncVectorEnv([lambda: make_env() for _ in range(10)])

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    actor = Actor(obs_dim, act_dim).to(device)
    target_actor = Actor(obs_dim, act_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())

    qf1 = QNetwork(obs_dim, act_dim).to(device)
    qf2 = QNetwork(obs_dim, act_dim).to(device)
    qf1_target = QNetwork(obs_dim, act_dim).to(device)
    qf2_target = QNetwork(obs_dim, act_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optim = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=config.learning_rate)
    actor_optim = optim.Adam(actor.parameters(), lr=config.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(config.buffer_size, envs.single_observation_space, envs.single_action_space, device)

    obs, _ = envs.reset(seed=config.seed)
    best_reward = -np.inf

    for step in range(config.total_timesteps):
        if step < config.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device)).cpu().numpy()
                actions += np.random.normal(0, config.exploration_noise, size=actions.shape)
                actions = actions.clip(envs.single_action_space.low, envs.single_action_space.high)

        args = get_config().parse_args()
        args.env_name = config.env_id
        args.noise_factor = config.noise_factor
        args.noise_type = config.noise_type
        args.llm_disturb_interval = config.llm_disturb_interval
        robust_input = [{"action": a, "robust_config": args} for a in actions]
        next_obs, rewards, terms, truncs, infos = envs.step(robust_input)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncs):
            if trunc and 'final_observation' in infos and infos['final_observation'] is not None:
                real_next_obs[idx] = infos['final_observation'][idx]

        # Extract per-env infos like in meow_robust
        per_env_infos = [{} for _ in range(envs.num_envs)]
        if 'final_observation' in infos:
            for i in range(envs.num_envs):
                per_env_infos[i]['final_observation'] = infos['final_observation'][i]
        if 'TimeLimit.truncated' in infos:
            for i in range(envs.num_envs):
                per_env_infos[i]['TimeLimit.truncated'] = infos['TimeLimit.truncated'][i]

        rb.add(obs, real_next_obs, actions, rewards, terms, per_env_infos)
        obs = next_obs

        if step >= config.learning_starts:
            data = rb.sample(config.batch_size)
            with torch.no_grad():
                noise = (torch.randn_like(data.actions) * config.policy_noise).clamp(-config.noise_clip, config.noise_clip)
                next_action = (target_actor(data.next_observations) + noise).clamp(-1.0, 1.0)
                target_q1 = qf1_target(data.next_observations, next_action)
                target_q2 = qf2_target(data.next_observations, next_action)
                target_q = data.rewards.flatten() + (1 - data.dones.flatten()) * config.gamma * torch.min(target_q1, target_q2).view(-1)

            q1 = qf1(data.observations, data.actions).view(-1)
            q2 = qf2(data.observations, data.actions).view(-1)
            q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            q_optim.zero_grad()
            q_loss.backward()
            q_optim.step()

            if step % config.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(config.tau * param.data + (1 - config.tau) * target_param.data)
                for q, q_target in zip([qf1, qf2], [qf1_target, qf2_target]):
                    for p, tp in zip(q.parameters(), q_target.parameters()):
                        tp.data.copy_(config.tau * p.data + (1 - config.tau) * tp.data)

            if step % 10000 == 0:
                avg_ret = evaluate(test_envs, actor, config, device)
                writer.add_scalar("return", avg_ret, step)
                if avg_ret > best_reward:
                    best_reward = avg_ret
                    torch.save(actor, os.path.join(config.description, 'best_policy.pt'))
                    print(f"step {step} -> {avg_ret:.2f} (best)")
                else:
                    print(f"step {step} -> {avg_ret:.2f}")

    envs.close()
    writer.close()

if __name__ == '__main__':
    config = Config.from_yaml("config_td3_robust.yaml")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    train(config)