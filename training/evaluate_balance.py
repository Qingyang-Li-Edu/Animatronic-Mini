"""Paired held-out pushes: learned standing versus neutral servo commands."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from balance_env import CharacterBalanceEnv, MODEL_PATH


def rollout(env, seed, force, policy=None, angle=None):
    options = {'push_force': force}
    if angle is not None:
        options['push_angle'] = angle
    obs, reset_info = env.reset(seed=seed, options=options)
    infos = []
    for _ in range(env.episode_steps):
        action = np.zeros(env.action_space.shape, dtype=np.float32) if policy is None else policy.predict(obs, deterministic=True)[0]
        obs, _, term, trunc, info = env.step(action)
        infos.append(info)
        if term or trunc:
            break
    final = infos[-1]
    success = (not term and final['tilt_deg'] < 5 and final['base_speed'] < .1 and
               final['drift_m'] < .05 and max(i['max_force_ratio'] for i in infos) < .7)
    return {'seed': seed, **reset_info, 'success': bool(success), 'terminated': bool(term),
            'duration_s': len(infos)*env.dt, 'rms_tilt_deg': float(np.sqrt(np.mean([i['tilt_deg']**2 for i in infos]))),
            'max_tilt_deg': max(i['tilt_deg'] for i in infos), 'final_tilt_deg': final['tilt_deg'],
            'final_base_speed': final['base_speed'], 'drift_m': final['drift_m'],
            'max_force_ratio': max(i['max_force_ratio'] for i in infos),
            'max_penetration_m': max(i['max_penetration_m'] for i in infos),
            'warnings': any(i['warnings'] for i in infos)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', type=Path)
    p.add_argument('--cases', type=int, default=30)
    p.add_argument('--forces', type=float, nargs='+', default=[0., .8, 3.])
    p.add_argument('--out', type=Path, default=Path('validation/reports/balance.json'))
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.cases < 1 or any(f < 0 for f in a.forces):
        p.error('Positive case count and nonnegative forces required')
    policy = PPO.load(a.policy, device='cpu') if a.policy else None
    env = CharacterBalanceEnv()
    report = {'scope': 'provisional free-root simulation only; no walking or real hardware',
              'policy': str(a.policy) if a.policy else None,
              'model_sha256': hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
              'criteria': '6s without fall/warnings; final tilt<5deg, base speed<.1, drift<.05m, force ratio<.7',
              'acceptance_force_n': .8, 'required_success_rate': .9,
              'seeds': [1000, 1000+a.cases-1], 'groups': []}
    for force in a.forces:
        for controller in (['neutral', 'ppo'] if policy else ['neutral']):
            rows = [rollout(env, 1000+i, force, policy if controller == 'ppo' else None)
                    for i in range(a.cases)]
            group = {'force_n': force, 'controller': controller,
                     'success_rate': float(np.mean([r['success'] for r in rows])),
                     'mean_rms_tilt_deg': float(np.mean([r['rms_tilt_deg'] for r in rows])),
                     'mean_duration_s': float(np.mean([r['duration_s'] for r in rows])),
                     'max_force_ratio': max(r['max_force_ratio'] for r in rows), 'cases': rows}
            report['groups'].append(group)
            print({k: v for k, v in group.items() if k != 'cases'}, flush=True)
    env.close()
    accepted = [g for g in report['groups'] if g['controller'] == 'ppo' and g['force_n'] == .8]
    report['status'] = ('passed' if accepted[0]['success_rate'] >= .9 else 'failed') if accepted else 'baseline_only'
    comparisons = []
    for force in a.forces:
        groups = {g['controller']: g for g in report['groups'] if g['force_n'] == force}
        if 'ppo' not in groups:
            continue
        baseline, trained = groups['neutral'], groups['ppo']
        comparisons.append({'force_n': force,
            'success_rate_difference': trained['success_rate']-baseline['success_rate'],
            'relative_rms_tilt_reduction': 1-trained['mean_rms_tilt_deg']/max(baseline['mean_rms_tilt_deg'], 1e-12),
            'note': 'Failed episodes end early; compare success first at high force'})
    report['paired_comparisons'] = comparisons
    # Keep task acceptance separate from evidence that learning adds value.
    report['learning_benefit_at_0_8n'] = any(c['force_n'] == .8 and
        c['relative_rms_tilt_reduction'] >= .1 and c['success_rate_difference'] >= 0
        for c in comparisons)
    strong = [g for g in report['groups'] if g['controller'] == 'ppo' and g['force_n'] == 3.]
    report['strong_push_status'] = ('passed' if strong[0]['success_rate'] >= .9 else 'failed') if strong else 'not_tested'
    if a.policy:
        report['policy_sha256'] = hashlib.sha256(a.policy.read_bytes()).hexdigest()
        report['training_manifest'] = json.loads(a.policy.with_suffix('.json').read_text())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(report['status'], a.out)
    raise SystemExit(1 if report['status'] == 'failed' else 0)


if __name__ == '__main__':
    main()
