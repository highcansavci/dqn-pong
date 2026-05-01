from env.make_env import make_env
from training.buffer import PrioritizedExperienceBuffer
from training.train import calc_loss
from training.agent import Agent

import time
import numpy as np

import torch
import torch.optim as optim
from dqn.dqn import DQN

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s -%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

from yaml import safe_load
from types import SimpleNamespace

config = safe_load(open('config/config.yaml', 'r'))


def dict_to_obj(d):
    return SimpleNamespace(**{k: dict_to_obj(v) if isinstance(v, dict) else v for k, v in d.items()})


config = dict_to_obj(config)

if __name__ == "__main__":
    device = torch.device(config.agent.device)
    env = make_env(config.env.name)
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    buffer = PrioritizedExperienceBuffer(config.agent.buffer_size, alpha=config.agent.per_alpha)
    agent = Agent(env, buffer, n_steps=config.agent.n_steps, gamma=config.agent.gamma)
    optimizer = optim.Adam(net.parameters(), lr=config.agent.lr)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None
    eps_span = config.agent.epsilon_start - config.agent.epsilon_end
    beta_span = 1.0 - config.agent.per_beta_start
    losses = []
    while True:
        frame_idx += 1
        epsilon = max(
            config.agent.epsilon_end,
            config.agent.epsilon_start - eps_span * frame_idx / config.agent.epsilon_decay_frames,
        )
        beta = min(
            1.0,
            config.agent.per_beta_start + beta_span * frame_idx / config.agent.per_beta_frames,
        )
        reward = agent.play_step(net, epsilon, device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            mean_loss = np.mean(losses) if losses else float('nan')
            losses.clear()
            mean_q = max_q = q_spread = float('nan')
            if len(buffer) >= config.agent.replay_initial_size:
                with torch.no_grad():
                    diag_states = buffer.sample(64, beta=beta)[0]
                    diag_states_v = torch.from_numpy(diag_states).to(device).float() / 255.0
                    diag_q = net(diag_states_v)
                    mean_q = diag_q.mean().item()
                    max_q = diag_q.max().item()
                    q_spread = diag_q.std(dim=1).mean().item()
            logging.info(
                f"Frame: {frame_idx}, Reward: {reward}, Mean Reward: {mean_reward:.2f}, "
                f"Loss: {mean_loss:.4f}, MeanQ: {mean_q:.3f}, MaxQ: {max_q:.3f}, "
                f"Qspread: {q_spread:.3f}, Beta: {beta:.2f}, Epsilon: {epsilon:.2f}, Speed: {speed:.2f} f/s"
            )
            if best_mean_reward is None or best_mean_reward < mean_reward:
                if best_mean_reward is not None:
                    logging.info(f"Best mean reward updated {best_mean_reward:.2f} -> {mean_reward:.2f}, model saved")
                best_mean_reward = mean_reward
                torch.save(net.state_dict(), config.agent.save_path)
            if mean_reward > config.env.mean_reward_threshold:
                logging.info(f"Solved in {frame_idx} frames!")
                break
        if len(buffer) < config.agent.replay_initial_size:
            continue

        if frame_idx % config.agent.target_update_freq == 0:
            tgt_net.load_state_dict(net.state_dict())

        if frame_idx % config.agent.train_freq != 0:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(config.agent.batch_size, beta=beta)
        loss_t, td_errors = calc_loss(batch, net, tgt_net, device)
        loss_t.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
        buffer.update_priorities(batch[5], td_errors)
        losses.append(loss_t.item())
