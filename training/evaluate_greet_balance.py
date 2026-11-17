"""Evaluate paired free-root greeting policies and a continuous-state run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
import torch
from stable_baselines3 import PPO

from greet_balance_env import GreetBalanceEnv, POSE_POLICY
from balance_env import MODEL_PATH


def rollout(env, policy, seed, force):
    obs, _ = env.reset(seed=seed, options={'push_force': force})
    rows, sliding, contact_slips = [], [], []
    previous = env.data.geom_xpos[env.feet].copy()
    for _ in range(env.episode_steps):
        action = policy.predict(obs, deterministic=True)[0] if policy else np.zeros(env.action_space.shape)
        obs, _, term, trunc, info = env.step(action)
        # Count tangential foot travel only while that sole has ground contact.
        contact_feet = set()
        for c in env.data.contact[:env.data.ncon]:
            if env.floor in (c.geom1, c.geom2):
                contact_feet.add(c.geom2 if c.geom1 == env.floor else c.geom1)
        slip = np.zeros(2)
        for c in env.data.contact[:env.data.ncon]:
            if env.floor not in (c.geom1, c.geom2):
                continue
            gid = c.geom2 if c.geom1 == env.floor else c.geom1
            if gid not in env.feet:
                continue
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_GEOM, int(gid), velocity, 0)
            point_velocity = velocity[3:] + np.cross(velocity[:3], c.pos-env.data.geom_xpos[gid])
            normal = np.asarray(c.frame).reshape(3,3)[0]
            tangent = point_velocity-normal*np.dot(point_velocity,normal)
            i = env.feet.index(gid)
            slip[i] = max(slip[i],float(np.linalg.norm(tangent))*env.dt)
        contact_slips.append(slip)
        now = env.data.geom_xpos[env.feet].copy()
        sliding.append([float(np.linalg.norm(now[i, :2]-previous[i, :2]))
                        if gid in contact_feet else 0. for i, gid in enumerate(env.feet)])
        previous = now
        rows.append(info)
        if term or trunc:
            break
    last = rows[-1]
    arm_rmse = float(np.sqrt(np.mean([r['arm_rmse_rad']**2 for r in rows])))
    slide = np.sum(sliding, axis=0)
    success = (not term and last['tilt_deg'] < 5 and last['base_speed'] < .1 and
               last['drift_m'] < .05 and max(r['max_force_ratio'] for r in rows) < .7 and arm_rmse < .05)
    return {'seed': seed, 'success': bool(success), 'terminated': bool(term),
            'duration_s': len(rows)*env.dt, 'rms_tilt_deg': float(np.sqrt(np.mean([r['tilt_deg']**2 for r in rows]))),
            'max_tilt_deg': max(r['tilt_deg'] for r in rows), 'final_tilt_deg': last['tilt_deg'],
            'final_base_speed': last['base_speed'], 'drift_m': last['drift_m'],
            'arm_rmse_rad': arm_rmse, 'grounded_foot_travel_m': slide.tolist(),
            'contact_point_slip_integral_m': np.sum(contact_slips,axis=0).tolist(),
            'max_force_ratio': max(r['max_force_ratio'] for r in rows),
            'max_penetration_m': max(r['max_penetration_m'] for r in rows),
            'warnings': any(r['warnings'] for r in rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', type=Path, default=Path('training/checkpoints/balance-seed7/stand.zip'))
    p.add_argument('--cases', type=int, default=30)
    p.add_argument('--baseline-policy', type=Path)
    p.add_argument('--forces', type=float, nargs='+', default=[0., .8, 3.])
    p.add_argument('--continuous-seconds', type=float, default=120.)
    p.add_argument('--out', type=Path, default=Path('validation/reports/greet_balance.json'))
    a = p.parse_args()
    torch.set_num_threads(1)
    policy = PPO.load(a.policy, device='cpu')
    old_policy = PPO.load(a.baseline_policy, device='cpu') if a.baseline_policy else None
    env = GreetBalanceEnv()
    report = {'scope': 'hierarchical free-root greeting simulation; uncalibrated physical parameters',
              'lower_policy': str(a.policy), 'upper_policy': str(POSE_POLICY),
              'model_sha256': hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
              'criteria': '8s (full greeting cycles plus quiet tail) without fall/warning; final tilt<5deg, base speed<.1, drift<.05m, arm RMSE<.05rad, force ratio<.7',
              'required_success_rate': .9, 'groups': []}
    for force in a.forces:
        controllers = {'neutral_legs':None, 'balance_ppo':policy}
        if old_policy is not None:
            controllers['previous_balance_ppo'] = old_policy
        for controller, selected in controllers.items():
            rows = [rollout(env, selected, 2000+i, force)
                    for i in range(a.cases)]
            group = {'force_n': force, 'controller': controller,
                     'success_rate': float(np.mean([r['success'] for r in rows])),
                     'mean_rms_tilt_deg': float(np.mean([r['rms_tilt_deg'] for r in rows])),
                     'worst_arm_rmse_rad': max(r['arm_rmse_rad'] for r in rows),
                     'max_grounded_foot_travel_m': max(max(r['grounded_foot_travel_m']) for r in rows),
                     'max_contact_slip_integral_m': max(max(r['contact_point_slip_integral_m']) for r in rows),
                     'cases': rows}
            report['groups'].append(group)
            print({k:v for k,v in group.items() if k!='cases'}, flush=True)
    env.close()
    if a.continuous_seconds > 0:
        env = GreetBalanceEnv(episode_seconds=a.continuous_seconds)
        # Extend pushes through the full run, keeping state continuous.
        original_reset = env.reset
        def reset(**kwargs):
            result = original_reset(**kwargs)
            env.push_times = tuple(np.arange(1., a.continuous_seconds-1., 1.5))
            return result
        env.reset = reset
        report['continuous'] = rollout(env, policy, 3000, .8)
        env.close()
        print('continuous', report['continuous'], flush=True)
    accepted = [g for g in report['groups'] if g['controller']=='balance_ppo' and g['force_n'] in (0., .8)]
    report['status'] = 'passed' if len(accepted)==2 and all(g['success_rate']>=.9 for g in accepted) and report.get('continuous',{}).get('success',False) else 'failed'
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(report['status'], a.out)
    raise SystemExit(0 if report['status']=='passed' else 1)


if __name__ == '__main__':
    main()
