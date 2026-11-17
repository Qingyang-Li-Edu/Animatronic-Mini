"""Continuous-state action switching test for the hierarchical controller."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO

from greet_balance_env import GreetBalanceEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', type=Path, default=Path('training/checkpoints/greet-balance-seed37/stand_greet.zip'))
    p.add_argument('--out', type=Path, default=Path('validation/reports/action_switching.json'))
    a = p.parse_args()
    torch.set_num_threads(1)
    policy = PPO.load(a.policy, device='cpu')
    env = GreetBalanceEnv(episode_seconds=32., push_force=.8)
    obs, _ = env.reset(seed=4040)
    phases = [('settle', 1., True), ('greet', 7., True), ('rest', 4., False),
              ('greet_again', 7., True), ('quiet', 5., False), ('greet_final', 8., True)]
    rows, failures, boundaries = [], [], []
    elapsed = 0.
    for name, seconds, upper_enabled in phases:
        env.set_upper_enabled(upper_enabled)
        count = round(seconds/env.dt)
        for index in range(count):
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            rows.append({'phase':name, 'time_s':elapsed, **info})
            if term:
                failures.append({'phase':name, 'time_s':elapsed, 'info':info})
                break
            elapsed += env.dt
        boundaries.append({'phase':name, 'time_s':elapsed, 'tilt_deg':rows[-1]['tilt_deg'],
                           'base_speed':rows[-1]['base_speed'], 'drift_m':rows[-1]['drift_m'],
                           'arm_rmse_rad':rows[-1]['arm_rmse_rad']})
        if failures:
            break
    active_rows = [r for r in rows if r['phase'] not in ('rest','quiet')]
    report = {'status':'passed' if not failures and max(r['arm_rmse_rad'] for r in active_rows) < .05 else 'failed',
              'policy':str(a.policy), 'continuous_seconds':elapsed,
              'phases':phases, 'boundaries':boundaries, 'failures':failures,
              'max_tilt_deg':max(r['tilt_deg'] for r in rows),
              'max_force_ratio':max(r['max_force_ratio'] for r in rows),
              'max_drift_m':max(r['drift_m'] for r in rows),
              'max_arm_rmse_rad':max(r['arm_rmse_rad'] for r in rows),
              'max_active_arm_rmse_rad':max(r['arm_rmse_rad'] for r in active_rows),
              'scope':'continuous MuJoCo state; upper-body command on/off transitions; no hardware validation'}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('boundaries','phases','failures')},indent=2))
    raise SystemExit(0 if report['status']=='passed' else 1)


if __name__ == '__main__':
    main()
