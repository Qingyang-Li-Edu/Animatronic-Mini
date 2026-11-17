#!/usr/bin/env python3
"""Torch-free baseline for pose tracking on macOS.

This is a behavior-cloning baseline, not PPO. It provides a deterministic
checkpoint and metrics while PyTorch/OpenMP packaging is resolved separately.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from pose_env import CharacterPoseEnv


def train(motion, samples, seed, output):
    env = CharacterPoseEnv(motion=motion)
    rng = np.random.default_rng(seed)
    observations, targets = [], []
    observation, _ = env.reset(seed=seed)
    for _ in range(samples):
        seconds = rng.uniform(0, env._duration())
        target = env._target(seconds)
        # The target is included in the observation by design. This baseline
        # learns a regularized actuator command from the full observation.
        observations.append(np.concatenate((env._obs(target), [1.0])))
        targets.append(target)
    x, y = np.asarray(observations), np.asarray(targets)
    ridge = 1e-3 * np.eye(x.shape[1])
    weights = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, weights=weights, motion=motion, seed=seed)
    errors = []
    observation, _ = env.reset(seed=seed)
    for _ in range(min(500, env.episode_steps)):
        action_target = env._target(env.step_count * env.frame_skip * env.model.opt.timestep)
        action = np.clip(np.asarray(np.concatenate((observation, [1.0])) @ weights),
                         env.ctrl_low, env.ctrl_high)
        # Convert actuator control to the environment's normalized action.
        action = 2 * (action - env.ctrl_low) / (env.ctrl_high - env.ctrl_low) - 1
        observation, _, terminated, truncated, info = env.step(action)
        errors.append(info['q_error_rms'])
        if terminated or truncated:
            break
    result = {'motion': motion, 'samples': samples, 'seed': seed,
              'mean_q_error_rms': float(np.mean(errors)),
              'max_q_error_rms': float(np.max(errors)),
              'steps': len(errors), 'checkpoint': str(output)}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion', default='greet')
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', type=Path, default=Path('training/checkpoints/linear/greet.npz'))
    args = parser.parse_args()
    train(args.motion, args.samples, args.seed, args.out)
