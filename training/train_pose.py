#!/usr/bin/env python3
"""Train a small PPO policy for fixed-base pose tracking."""
import argparse
from pathlib import Path
from functools import partial
import hashlib
import importlib.metadata
import json
import platform
import time

from pose_env import CharacterPoseEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion', default='greet', choices=['nod', 'shake', 'look_around', 'crouch', 'greet', 'blink'])
    parser.add_argument('--timesteps', type=int, default=100_000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', default='training/checkpoints/pose')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--action-mode',choices=['absolute','residual'],default='residual')
    parser.add_argument('--force-cap',type=float,default=.65)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        env = CharacterPoseEnv(motion=args.motion,action_mode=args.action_mode,force_cap_ratio=args.force_cap)
        observation, info = env.reset(seed=7)
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, metrics = env.step(action)
        print({'obs': observation.shape, 'next_obs': next_observation.shape,
               'action': env.action_space.shape, 'reward': reward,
               'terminated': terminated, 'truncated': truncated, 'metrics': metrics})
        return
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
        import torch
    except ImportError as exc:
        raise SystemExit('Install stable-baselines3, gymnasium, and torch first') from exc
    if args.workers < 1 or args.timesteps < 1:
        raise SystemExit('workers and timesteps must be positive')
    torch.set_num_threads(1)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS is not available in this process; use CPU')
    factories = [partial(CharacterPoseEnv, motion=args.motion,action_mode=args.action_mode,
                         force_cap_ratio=args.force_cap) for _ in range(args.workers)]
    # DummyVecEnv is sequential; use actual spawn workers for parallel MuJoCo.
    vector = DummyVecEnv(factories) if args.workers == 1 else SubprocVecEnv(factories, start_method='spawn')
    env = VecMonitor(vector)
    output = Path(args.out) / args.motion
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = PPO('MlpPolicy', env, n_steps=256, batch_size=256, learning_rate=3e-4,
                 gamma=.99, gae_lambda=.95, ent_coef=0.0, verbose=1,
                 device=args.device, seed=args.seed, tensorboard_log=str(output.parent / 'tensorboard'),
                 policy_kwargs={'log_std_init':-1.5})
    from pose_env import MODEL_PATH, MOTIONS_PATH
    manifest = {'algorithm':'PPO', 'motion':args.motion, 'seed':args.seed,
                'workers':args.workers, 'device':args.device, 'requested_steps':args.timesteps,
                'action_mode':args.action_mode,'residual_scale':.12,'force_cap_ratio':args.force_cap,
                'python':platform.python_version(), 'platform':platform.machine(),
                'packages':{p:importlib.metadata.version(p) for p in
                            ('torch','mujoco','numpy','gymnasium','stable-baselines3')},
                'model_sha256':hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                'motions_sha256':hashlib.sha256(MOTIONS_PATH.read_bytes()).hexdigest()}
    start = time.monotonic()
    try:
        policy.learn(total_timesteps=args.timesteps,log_interval=25)
        policy.save(str(output))
        manifest.update(completed_steps=policy.num_timesteps, elapsed_seconds=time.monotonic()-start)
        output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    finally:
        env.close()
    print('Saved', output)


if __name__ == '__main__':
    main()
