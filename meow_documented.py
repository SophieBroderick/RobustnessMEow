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
    learning_rate: float
    noise_clip: float
    alpha: float
    grad_clip: float
    log_sigma_max: float
    log_sigma_min: float
    description: str = ''

    @staticmethod
    def from_yaml(path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return Config(**data)

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
            new_obs, rewards, terminated, truncated, _ = envs.step(act)
            done = terminated | truncated
            returns += rewards * (1-dones)
            dones |= done
            obs = new_obs
    return returns.mean()

def train(config: Config):

    run_name = f"{config.env_id}__meow__{config.seed}__{datetime.now().strftime(r'%y%m%d_%H%M%S')}"
    config.description = f'runs/{run_name}'
    writer = SummaryWriter(f'runs/{run_name}')

    # Torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup the training and testing environment
    def make_env(env_id, seed):
        def thunk():
            env = gym.make(env_id)
            env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
            env.action_space.seed(seed)
            return env
        return thunk
    train_envs = gym.vector.SyncVectorEnv([make_env(config.env_id, config.seed)])
    test_envs = gym.make_vec(config.env_id, num_envs=10)
    test_envs = gym.wrappers.RescaleAction(test_envs, min_action=-1.0, max_action=1.0)

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
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)

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
                next_obs, rewards, terminations, truncations, infos = train_envs.step(actions)

                ''' Step 5 - Add the new SARS to the replay buffer '''

                # Update replay buffer (handling truncations)
                real_next_obs = next_obs.copy()
                for idx, trunc in enumerate(truncations):
                    if trunc:
                        real_next_obs[idx] = infos["final_observation"][idx]
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
                    plt.xlim(left=0, right=config.total_timesteps)
                    plt.ylim(bottom=0)
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

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Create the configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    config = Config.from_yaml(parser.parse_args().config)

    # Random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    # Training
    train(config)
