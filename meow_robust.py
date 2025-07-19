import os
import sys

import random
import yaml
import argparse
from dataclasses import dataclass
from tqdm.auto import tqdm
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import gymnasium as gym
import robust_gymnasium as robust_gym
from robust_gymnasium.configs.robust_setting import get_config

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from ebflows.flow_policy import FlowPolicy

@dataclass
class Config:
    seed: int
    env_id: str
    total_timesteps: int
    buffer_size: int
    gamma: float
    tau: float
    batch_size: int
    learning_start_step: int
    q_lr: float
    noise_clip: float
    alpha: float
    grad_clip: float
    log_sigma_max: float
    log_sigma_min: float
    description: str = ''
    noise_factor: str = None
    noise_type: str = None
    llm_disturb_interval: int = None

    @staticmethod
    def from_yaml(path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return Config(**data)

# Action rescaling for the Robustness Gym format
class RobustRescaleActionWrapper:
    def __init__(self, env, min_action=-1.0, max_action=1.0):
        self.env = env

        true_action_space = getattr(env, 'unwrapped', env).action_space
        required_attrs = ['low', 'high', 'shape']
        if not all(hasattr(true_action_space, attr) for attr in required_attrs):
            raise TypeError(f"Expected Box-like action space with low/high/shape, got: {type(true_action_space)}")

        self.orig_low = true_action_space.low
        self.orig_high = true_action_space.high
        self.min_action = min_action
        self.max_action = max_action

        self.action_space = gym.spaces.Box(
            low=min_action, high=max_action, shape=true_action_space.shape, dtype=np.float32
        )

        obs_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.array(obs_space.low, dtype=np.float32),
            high=np.array(obs_space.high, dtype=np.float32),
            shape=obs_space.shape,
            dtype=np.float32
        )
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action_input):
        if isinstance(action_input, dict):
            raw_action = action_input["action"]
            config = action_input.get("robust_config", None)
        else:
            raw_action = action_input
            config = None

        scaled_action = self.orig_low + (0.5 * (raw_action + 1.0) * (self.orig_high - self.orig_low))
        input_dict = {"action": scaled_action}
        if config is not None:
            input_dict["robust_config"] = config

        return self.env.step(input_dict)

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)

def eval_test_returns(envs, policy):
    with torch.no_grad():
        policy.eval()
        num_envs = envs.unwrapped.num_envs
        returns = np.zeros((num_envs,))
        dones = np.zeros((num_envs,)).astype(bool)
        obs, _ = envs.reset(seed=range(num_envs))
        while not all(dones):
            act, _ = policy.sample(num_samples=num_envs, obs=obs, deterministic=True)
            act = act.cpu().detach().numpy()

            args = get_config().parse_args()
            #args.env_name = "Ant-v4"
            #args.noise_factor = "reward"
            #args.noise_type = "gauss"
            #args.llm_disturb_interval = 500
            robust_input = [{"action": a, "robust_config": args} for a in act]

            new_obs, rewards, terminated, truncated, _ = envs.step(robust_input)
            done = terminated | truncated
            returns += rewards * (1-dones)
            dones |= done
            obs = new_obs
    return returns.mean()

def train(config):

    run_name = f'{config.env_id}_meow_{config.seed}_{config.noise_factor}_{datetime.now().strftime('%y%m%d_%H%M%S')}'
    config.description = f'runs/{run_name}'
    writer = SummaryWriter(f'runs/{run_name}')

    # Torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup the training and testing environment
    def make_train_env(env_id, seed):
        def thunk():
            env = robust_gym.make(env_id)
            env = RobustRescaleActionWrapper(env, min_action=-1.0, max_action=1.0)
            env.action_space.seed(seed)
            return env
        return thunk
    train_envs = gym.vector.SyncVectorEnv([make_train_env(config.env_id, config.seed)])

    # Set up the testing environment
    def make_test_env(env_id):
        def thunk():
            env = robust_gym.make(env_id)
            env = RobustRescaleActionWrapper(env, min_action=-1.0, max_action=1.0)
            return env
        return thunk
    test_envs = gym.vector.SyncVectorEnv([make_test_env(config.env_id) for _ in range(10)])

    # Flow-based policy
    policy = FlowPolicy(
        alpha=config.alpha,
        log_sigma_max=config.log_sigma_max,
        log_sigma_min=config.log_sigma_min,
        action_sizes=train_envs.action_space.shape[1],
        state_sizes=train_envs.observation_space.shape[1],
        device=device
    ).to(device)

    # Flow-based target policy
    policy_target = FlowPolicy(
        alpha=config.alpha,
        log_sigma_max=config.log_sigma_max,
        log_sigma_min=config.log_sigma_min,
        action_sizes=train_envs.action_space.shape[1],
        state_sizes=train_envs.observation_space.shape[1],
        device=device
    ).to(device)
    policy_target.load_state_dict(policy.state_dict())
    policy_target.eval()

    # One optimizer (for both actor and critic)
    optimizer = optim.Adam(policy.parameters(), lr=config.q_lr)

    # Create a replay buffer
    train_envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        config.buffer_size,
        train_envs.single_observation_space,
        train_envs.single_action_space,
        device,
        handle_timeout_termination=False
    )

    best_test_return = -np.inf
    plot_steps = []
    plot_returns = []

    # First step
    obs, _ = train_envs.reset(seed=config.seed)

    # Progress bar, iterate over all steps
    for global_step_stride in range(config.total_timesteps // 10000):
        with tqdm(range(10000), position=0, leave=False, bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
            for global_step_mod in pbar:
                global_step = global_step_stride * 10000 + global_step_mod

                ''' |> Step 2 - Extend the replay buffer '''

                ''' Step 3 - Sample actions for the replay buffer '''

                # Decide if learning should start yet, and sample actions accordingly
                if global_step < config.learning_start_step:
                    actions = np.array([train_envs.single_action_space.sample() for _ in range(train_envs.num_envs)])
                else:
                    policy.eval()
                    actions, _ = policy.sample(num_samples=obs.shape[0], obs=obs, deterministic=False)
                    actions = actions.detach().cpu().numpy()

                ''' Step 4 - Sample observations for the replay buffer '''

                # Environment step
                args = get_config().parse_args()
                args.env_name = "Ant-v4"
                args.noise_factor = "state"
                args.noise_type = "gauss"
                args.llm_disturb_interval = 500
                robust_input = [{"action": a, "robust_config": args} for a in actions]
                next_obs, rewards, terminations, truncations, infos = train_envs.step(robust_input)

                ''' Step 5 - Add the new SARS to the replay buffer '''

                # Update replay buffer (handling truncations)
                real_next_obs = next_obs.copy()
                for idx, trunc in enumerate(truncations):
                    if trunc and 'final_observation' in infos and infos['final_observation'] is not None:
                        real_next_obs[idx] = infos['final_observation'][idx]
                rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

                # Update with new observation
                obs = next_obs

                # Skip learning from early iterations (for stability)
                if global_step <= config.learning_start_step:
                    continue

                ''' |> Step 6 - Update policy '''

                ''' Step 7 - Sample a SARS set from the replay buffer '''

                # Get the target value estimate for updates
                data = rb.sample(config.batch_size)
                rb_obs = data.observations
                rb_act = data.actions
                rb_rew = data.rewards
                rb_next_obs = data.next_observations
                rb_done = data.dones

                ''' Steps 8 to 11 - Calculate necessary components for the loss '''

                # Calculate the target Q
                with torch.no_grad():
                    v_old = policy_target.get_v(torch.cat((rb_next_obs, rb_next_obs), dim=0))
                    exact_v_old = torch.min(v_old[:v_old.shape[0]//2], v_old[v_old.shape[0]//2:])
                    target_q = rb_rew.flatten() + (1-rb_done.flatten()) * config.gamma * (exact_v_old).view(-1)

                # Put the policy in training mode
                policy.train()
                current_q1, _ = policy.get_qv(
                    torch.cat((rb_obs, rb_obs), dim=0),
                    torch.cat((rb_act, rb_act), dim=0)
                )
                target_q = torch.cat((target_q, target_q), dim=0)

                ''' Step 12 - Compute the loss (with just Q1, not Q2) '''

                # Get the mean of the MSE loss for each environment, removing NaNs
                loss = F.mse_loss(current_q1.flatten(), target_q.flatten())
                loss[loss != loss] = 0.0
                loss = loss.mean()

                ''' Step 13 - Update parameters '''

                # Train the actor and critic simultaneously
                optimizer.zero_grad()
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
                optimizer.step()

                ''' Step 14 - Target policy update '''

                # Target network update for the policy
                for param, target_param in zip(policy.parameters(), policy_target.parameters()):
                    target_param.data.copy_(config.tau * param.data + (1 - config.tau) * target_param.data)

                ''' Evaluation '''

                # Evaluation
                if global_step % 10000 == 0:
                    sys.stdout.write('\x1b[2K\r')
                    test_return = eval_test_returns(test_envs, policy)
                    writer.add_scalar('return', test_return, global_step)
                    if test_return > best_test_return:
                        best_test_return = test_return
                        torch.save(policy, os.path.join(f'{config.description}', 'best_policy.pt'))
                        print(f'\033[92mstep {global_step: 7d} -> {test_return:.2f}\033[0m')
                    else:
                        print(f'step {global_step: 7d} -> {test_return:.2f}')
                    plot_steps.append(global_step)
                    plot_returns.append(test_return)
                    plt.clf()
                    plt.figure(figsize=(8, 6), dpi=1000)
                    sns.lineplot(x=plot_steps, y=plot_returns, label='MEow', color='blue')
                    plt.xlim(left=0, right=2500000)
                    plt.title(f'Test Returns for {config.env_id}')
                    plt.xlabel('Training Step')
                    plt.ylabel('Return')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(f'{config.description}', 'returns.png'))
                    plt.close()

            pbar.close()

    train_envs.close()
    writer.close()

if __name__ == '__main__':

    # Create the configuration
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--config')
    #args, _ = parser.parse_known_args()
    #config = Config.from_yaml(args.config)
    
    #config = Config.from_yaml(config_path)
    config = Config.from_yaml("config_meow_robust.yaml")

    # Random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    # Training
    train(config)