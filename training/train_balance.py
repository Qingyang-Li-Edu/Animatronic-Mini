"""Train a free-root standing controller on CPU with actual spawn workers."""
import argparse
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from balance_env import CharacterBalanceEnv, MODEL_PATH


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=524288)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--push-force', type=float, default=.8)
    p.add_argument('--out', type=Path, default=Path('training/checkpoints/balance'))
    a = p.parse_args()
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    env = VecMonitor(SubprocVecEnv([partial(CharacterBalanceEnv, push_force=a.push_force)
                                   for _ in range(2)], start_method='spawn'))
    policy = PPO('MlpPolicy', env, n_steps=512, batch_size=256, learning_rate=3e-4,
                 gamma=.995, gae_lambda=.95, ent_coef=0., seed=a.seed, device='cpu',
                 policy_kwargs={'log_std_init': -2.}, verbose=1,
                 tensorboard_log=str(a.out/'tensorboard'))
    start = time.monotonic()
    try:
        policy.learn(a.timesteps, log_interval=25)
        policy.save(a.out/'stand')
        manifest = {'algorithm': 'PPO', 'environment_version': CharacterBalanceEnv.version,
                    'seed': a.seed, 'push_force_n': a.push_force,
                    'push_duration_s': .12, 'push_times_s': [1., 2.5, 4.],
                    'completed_steps': policy.num_timesteps, 'workers': 2, 'device': 'cpu',
                    'action_scale_rad': .12, 'force_cap_ratio': .65,
                    'elapsed_seconds': time.monotonic()-start,
                    'model_sha256': hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                    'environment_sha256': hashlib.sha256(Path(__file__).with_name('balance_env.py').read_bytes()).hexdigest(),
                    'python': platform.python_version(),
                    'packages': {n: importlib.metadata.version(n) for n in
                                 ('torch', 'mujoco', 'numpy', 'gymnasium', 'stable-baselines3')},
                    'randomization': 'independent body mass/inertia x0.8..1.2, friction .5..1.0, initial tilt +/-2deg, joint q sigma .005rad, qvel sigma .005',
                    'scope': 'provisional free-root standing simulation; no walking or hardware validation'}
        (a.out/'stand.json').write_text(json.dumps(manifest, indent=2)+'\n')
    finally:
        env.close()
    print('Saved', a.out/'stand.zip')


if __name__ == '__main__':
    main()
