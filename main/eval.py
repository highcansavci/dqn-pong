import argparse
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import ale_py

from wrappers.max_and_skip_env import MaxAndSkipEnv
from wrappers.warp_frame import WarpFrame
from wrappers.buffer_wrapper import BufferWrapper
from dqn.dqn import DQN

from yaml import safe_load

gym.register_envs(ale_py)


def dict_to_obj(d):
    return SimpleNamespace(**{k: dict_to_obj(v) if isinstance(v, dict) else v for k, v in d.items()})


def make_eval_env(env_id, video_dir=None, name_prefix="dqn-eval"):
    env = gym.make(env_id, render_mode="rgb_array")
    if video_dir is not None:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda ep_id: True,
            name_prefix=name_prefix,
            disable_logger=True,
        )
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = BufferWrapper(env, n_steps=4)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/dqn.pth")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--video-dir", default="videos")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = dict_to_obj(safe_load(open("config/config.yaml", "r")))
    device = torch.device(args.device or config.agent.device)

    video_dir = None if args.no_video else Path(args.video_dir)
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    env = make_eval_env(config.env.name, video_dir=video_dir)
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    net.eval()

    rng = np.random.default_rng(args.seed)
    reset_kwargs = {"seed": args.seed} if args.seed is not None else {}

    rewards = []
    for ep in range(args.episodes):
        state, _ = env.reset(**reset_kwargs)
        reset_kwargs = {}  # only seed the first episode
        total_reward = 0.0
        steps = 0
        while True:
            if rng.random() < args.epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_v = torch.from_numpy(state[None]).to(device).float() / 255.0
                    action = int(net(state_v).argmax(1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        rewards.append(total_reward)
        print(f"Episode {ep + 1}/{args.episodes}: reward={total_reward:+.0f}, steps={steps}")

    env.close()
    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    print(f"\nMean reward over {args.episodes} episodes: {mean:+.2f} ± {std:.2f}")
    if video_dir is not None:
        print(f"Videos saved to {video_dir.resolve()}")


if __name__ == "__main__":
    main()
