# DQN Pong

A Deep Q-Network agent that learns to play Atari Pong (`PongNoFrameskip-v4`) from raw pixels.

## Demo

Trained agent playing at evaluation epsilon = 0.01, scoring **21-0**:

[![demo](https://img.shields.io/badge/Watch_demo-21--0_win-2ea44f?logo=github)](https://github.com/highcansavci/dqn-pong/blob/main/videos/dqn-eval-episode-0.mp4)

Direct file: [videos/dqn-eval-episode-0.mp4](videos/dqn-eval-episode-0.mp4)

## Results

- Solved in ~4M environment-loop frames
- Evaluation mean reward over 5 episodes at epsilon = 0.01: **+18.2** (range +16 to +21)
- Training throughput: ~250 frames/sec on a single GPU

## Features

- Standard Atari preprocessing: grayscale, 84x84 resize, 4-frame stack, frame-skip 4 with max-pool over the last 2 frames
- uint8 replay buffer (~28 KB per state stack) — 100k transitions fit in ~5.6 GB RAM
- **Prioritized Experience Replay** (proportional, sum-tree backed)
- **n-step returns** (n = 3)
- **Double DQN** target
- **Huber loss** + gradient clipping at norm 10
- Linear epsilon schedule: 1.0 -> 0.1 over 1M frames
- Diagnostic logging: per-state Q-spread, mean Q, max Q, loss, beta

## Setup

```bash
pip install gymnasium ale-py autorom torch opencv-python pyyaml moviepy
AutoROM --accept-license
```

## Usage

Train:
```bash
python -m main.main
```

Evaluate (saves mp4 recordings of each episode to `videos/`):
```bash
python -m main.eval --episodes 5
python -m main.eval --episodes 30 --no-video    # quick benchmark, no recording
python -m main.eval --epsilon 0 --seed 42       # deterministic single-run
```

## Project structure

```
config/         # YAML hyperparameters
dqn/            # CNN architecture (Conv8s4-32 -> Conv4s2-64 -> Conv3s1-64 -> FC512 -> n_actions)
env/            # Environment factory
main/           # Train + eval entrypoints
training/       # Agent, replay buffer (PER + sum tree), loss
wrappers/       # Atari preprocessing wrappers
checkpoints/    # Saved network weights
videos/         # Eval episode recordings
```

## Hyperparameters

Defined in `config/config.yaml`. Key values:

| Setting | Value |
|---|---|
| Learning rate | 0.00025 (Adam) |
| Batch size | 128 |
| Buffer size | 100,000 |
| Replay initial size | 10,000 |
| Target update freq | 1000 frames |
| Train freq | every 4 frames |
| n-step | 3 |
| Discount gamma | 0.99 |
| PER alpha | 0.6 |
| PER beta | 0.4 -> 1.0 over 1M frames |
| Epsilon | 1.0 -> 0.1 over 1M frames |

## Notes

The replay buffer stores frames as uint8; division by 255 happens on the GPU after sampling. This keeps the buffer ~4x smaller than the float32 alternative, which matters because Atari preprocessing is the single largest factor in DQN training throughput.
