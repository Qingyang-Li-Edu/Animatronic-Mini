"""Fine-tune leg balance while the fixed-base pose PPO drives the upper body."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from greet_balance_env import GreetBalanceEnv, POSE_POLICY
from balance_env import MODEL_PATH


def factory(force):
    torch.set_num_threads(1)
    return GreetBalanceEnv(push_force=force)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=262144)
    p.add_argument('--seed', type=int, default=37)
    p.add_argument('--push-force', type=float, default=.8)
    p.add_argument('--init', type=Path, default=Path('training/checkpoints/balance-seed7/stand.zip'))
    p.add_argument('--out', type=Path, default=Path('training/checkpoints/greet-balance-seed37'))
    a = p.parse_args()
    torch.set_num_threads(1)
    env = VecMonitor(SubprocVecEnv([partial(factory, a.push_force) for _ in range(2)], start_method='spawn'))
    model = PPO.load(a.init, env=env, device='cpu', learning_rate=1e-4,
                     tensorboard_log=str(a.out/'tensorboard'), seed=a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        model.learn(a.timesteps, reset_num_timesteps=True, log_interval=25)
        model.save(a.out/'stand_greet')
        manifest = {'algorithm': 'hierarchical PPO; frozen upper-body pose PPO, fine-tuned leg PPO',
                    'seed': a.seed, 'push_force_n': a.push_force, 'workers': 2, 'device': 'cpu',
                    'completed_steps': model.num_timesteps, 'init': str(a.init),
                    'upper_policy': str(POSE_POLICY), 'elapsed_seconds': time.monotonic()-start,
                    'model_sha256': hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                    'upper_policy_sha256': hashlib.sha256(POSE_POLICY.read_bytes()).hexdigest(),
                    'environment_sha256': hashlib.sha256(Path(__file__).with_name('greet_balance_env.py').read_bytes()).hexdigest(),
                    'episode_seconds': 8., 'scope': 'free-root greeting compensation, no locomotion/hardware calibration'}
        (a.out/'stand_greet.json').write_text(json.dumps(manifest, indent=2)+'\n')
    finally:
        env.close()
    print('Saved', a.out/'stand_greet.zip')


if __name__ == '__main__':
    main()
