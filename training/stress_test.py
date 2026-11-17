#!/usr/bin/env python3
"""Repeat learned pose tracking and test held-out simulation perturbations."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from stable_baselines3 import PPO
import torch
from pose_env import CharacterPoseEnv


def episode(env, policy, obs):
    errors, ratios = [], []
    for _ in range(env.episode_steps):
        action,_=policy.predict(obs,deterministic=True)
        obs,_,term,trunc,info=env.step(action)
        assert not term and np.isfinite(obs).all(), 'nonfinite rollout'
        errors.append(info['active_q_error_rms'])
        ratios.append(float(np.max(np.abs(env.data.actuator_force)/env.configured_force_limits)))
    return obs, {'active_joint_rmse_rad':float(np.sqrt(np.mean(np.square(errors)))),
                 'max_force_ratio':max(ratios)}


def test(path,repeats=100):
    torch.set_num_threads(1)
    manifest=json.loads(path.with_suffix('.json').read_text())
    policy=PPO.load(path,device='cpu')
    def factory():
        return CharacterPoseEnv(motion=manifest['motion'],action_mode=manifest['action_mode'],
                                force_cap_ratio=manifest['force_cap_ratio'])
    env=factory(); obs,_=env.reset(seed=200)
    cycles=[]
    for n in range(repeats):
        if n:
            env.step_count=0
            obs=env._obs(env._target(0))
        obs,result=episode(env,policy,obs)
        cycles.append(result)
    random_cases=[]
    for seed in range(100,110):
        env=factory(); rng=np.random.default_rng(seed)
        scales=rng.uniform(.8,1.2,size=(env.model.nbody-1,1))
        env.model.body_mass[1:]*=scales[:,0]
        env.model.body_inertia[1:]*=scales
        mujoco.mj_setConst(env.model,env.data)
        obs,_=env.reset(seed=seed)
        for j in range(env.model.njnt):
            adr=env.model.jnt_qposadr[j]
            env.data.qpos[adr]=np.clip(rng.normal(0,.005),*env.model.jnt_range[j])
        env.data.qvel[:]=rng.normal(0,.01,size=env.model.nv)
        mujoco.mj_forward(env.model,env.data)
        obs=env._obs(env._target(0))
        _,result=episode(env,policy,obs)
        random_cases.append({'seed':seed,**result})
    summary={'repeat_count':repeats,
             'repeat_worst_active_rmse_rad':max(x['active_joint_rmse_rad'] for x in cycles),
             'randomized_cases':len(random_cases),
             'randomized_worst_active_rmse_rad':max(x['active_joint_rmse_rad'] for x in random_cases),
             'max_force_ratio':max(x['max_force_ratio'] for x in cycles+random_cases)}
    passed=(summary['repeat_worst_active_rmse_rad']<.05 and
            summary['randomized_worst_active_rmse_rad']<.05 and summary['max_force_ratio']<.7)
    return {'status':'passed' if passed else 'failed','summary':summary,
            'policy':str(path),'randomization':'independent body mass/inertia x0.8..1.2; initial q sigma .005 rad, qvel sigma .01 rad/s',
            'scope':'fixed-base tracking robustness only; not a balance or locomotion test',
            'continuous_cycles':cycles,'randomized_rollouts':random_cases}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--out',type=Path,default=Path('validation/reports/greet_stress.json'))
    a=p.parse_args();r=test(a.policy,a.repeats)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r['summary'],indent=2)); print(r['status'],a.out)
    raise SystemExit(0 if r['status']=='passed' else 1)
