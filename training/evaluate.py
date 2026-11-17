#!/usr/bin/env python3
"""Compare a trained policy with neutral and direct-reference controllers."""
import argparse
from pathlib import Path
import json
import numpy as np
from pose_env import CharacterPoseEnv


def rollout(motion, controller, action_mode='absolute',force_cap_ratio=1.):
    env = CharacterPoseEnv(motion=motion,action_mode=action_mode,force_cap_ratio=force_cap_ratio)
    obs, _ = env.reset(seed=101)
    error, peak, torques = [], [], []
    reward_sum = 0.
    for _ in range(env.episode_steps):
        if controller == 'reference':
            target = env._target(env.step_count*env.frame_skip*env.model.opt.timestep)
            action = 2*(target-env.ctrl_low)/(env.ctrl_high-env.ctrl_low)-1
        elif controller == 'neutral':
            action = 2*(np.zeros(env.model.nu)-env.ctrl_low)/(env.ctrl_high-env.ctrl_low)-1
        else:
            action, _ = controller.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        error.append(info['active_q_error_rms'])
        peak.append(info['q_error_rms'])
        limits = env.configured_force_limits
        torques.append(np.abs(env.data.actuator_force)/limits)
        reward_sum += reward
        if terminated or truncated:
            break
    return {'active_joint_rmse_rad':float(np.sqrt(np.mean(np.square(error)))),
            'all_joint_rmse_rad':float(np.sqrt(np.mean(np.square(peak)))),
            'max_actuator_force_ratio':float(np.max(torques)),
            'return':reward_sum, 'steps':len(error), 'finite':bool(np.isfinite(obs).all()),
            'scope':'single deterministic nominal rollout; robustness not yet evaluated'}


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--motion',default='greet')
    p.add_argument('--policy',type=Path)
    p.add_argument('--out',type=Path,default=Path('validation/reports/pose_evaluation.json'))
    args=p.parse_args()
    results={name:rollout(args.motion,name) for name in ('neutral','reference')}
    if args.policy:
        from stable_baselines3 import PPO
        import torch
        torch.set_num_threads(1)
        policy=PPO.load(args.policy,device='cpu')
        manifest=args.policy.with_suffix('.json')
        configuration=json.loads(manifest.read_text()) if manifest.is_file() else {}
        results['ppo']=rollout(args.motion,policy,configuration.get('action_mode','absolute'),
                               configuration.get('force_cap_ratio',1.))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'motion':args.motion,'results':results},indent=2)+'\n')
    print(json.dumps(results,indent=2))
