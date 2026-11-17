"""Stream actual local MuJoCo policy states to the downloaded VRM shell."""
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))


class TrainingBridge:
    def __init__(self):
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.pending = 'balance'
        self.state = {}
        self.received_at = 0.
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def set_motion(self, name):
        with self.lock:
            self.pending = name

    def snapshot(self):
        with self.lock:
            age = time.monotonic()-self.received_at if self.received_at else None
            return {**self.state, 'connected': age is not None and age < 2 and not self.error,
                    'age_seconds': age, 'error': self.error,
                    'source': 'local_mujoco_training', 'joint_count': len(self.state.get('positions', {}))}

    def run(self):
        try:
            import mujoco
            import numpy as np
            import torch
            from stable_baselines3 import PPO
            from balance_env import CharacterBalanceEnv
            from pose_env import CharacterPoseEnv
            from greet_balance_env import GreetBalanceEnv
            import json
            torch.set_num_threads(1)
            policies = {}
            env, motion, policy, episode = None, None, None, 0
            deadline = time.monotonic()
            while not self.stopped.is_set():
                with self.lock:
                    requested, self.pending = self.pending, None
                if requested is not None:
                    if env is not None:
                        env.close()
                    motion, episode = requested, 0
                    if motion in ('greet_balance', 'greet_balance_strong'):
                        accepted = ROOT/'validation/reports/greet_balance_seed37.json'
                        if not accepted.exists() or json.loads(accepted.read_text())['status'] != 'passed':
                            raise RuntimeError('Standing greeting has not passed validation')
                        env = GreetBalanceEnv(push_force=.8 if motion=='greet_balance' else 3.)
                        key = 'greet-balance-seed37/stand_greet'
                    elif motion in ('balance', 'balance_strong'):
                        env = CharacterBalanceEnv(push_force=.8 if motion == 'balance' else 3.)
                        key = 'balance-seed7/stand'
                    else:
                        env = CharacterPoseEnv(motion=motion, action_mode='residual', force_cap_ratio=.65)
                        key = 'ppo-safe/greet' if motion == 'greet' else None
                    if key and key not in policies:
                        policies[key] = PPO.load(ROOT/'training/checkpoints'/f'{key}.zip', device='cpu')
                    policy = policies.get(key)
                    obs, _ = env.reset(seed=1000)
                    deadline = time.monotonic()
                if env is None:
                    self.stopped.wait(.05)
                    continue
                action = policy.predict(obs, deterministic=True)[0] if policy else np.zeros(env.action_space.shape)
                # Nontrained motion buttons send the original reference command,
                # not a fabricated animation. Residual zero is direct reference.
                obs, _, terminated, truncated, info = env.step(action)
                base = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'base')
                positions = {mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, int(j)):
                    float(env.data.qpos[env.model.jnt_qposadr[j]])
                    for j in env.model.actuator_trnid[:, 0]}
                free = motion in ('balance', 'balance_strong', 'greet_balance', 'greet_balance_strong')
                feet = {s:env.data.xpos[mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,s+'_foot')].tolist()
                        for s in ('left','right')}
                state = {'positions': positions, 'mode': motion, 'episode': episode,
                         'simulation_time': float(env.data.time), 'controller': 'PPO' if policy else 'reference',
                         'root': {'position': env.data.xpos[base].tolist(),
                                  'quaternion_wxyz': env.data.xquat[base].tolist(),
                                  'reference_position': [-.000291, 0., .156785] if free else [0., 0., .17],
                                  'free': free},
                         'physics_feet':feet,
                         'push_force_n': float(np.linalg.norm(env.data.xfrc_applied[base, :3])),
                         'metrics': info, 'episode_ended': bool(terminated or truncated)}
                with self.lock:
                    self.state, self.received_at = state, time.monotonic()
                dt = env.frame_skip*env.model.opt.timestep
                deadline += dt
                self.stopped.wait(max(0., deadline-time.monotonic()))
                if terminated or truncated:
                    # Briefly hold a genuine end state so a fall is visible.
                    if terminated:
                        self.stopped.wait(1.)
                    episode += 1
                    obs, _ = env.reset(seed=1000+episode)
            if env is not None:
                env.close()
        except Exception as error:
            with self.lock:
                self.error = f'{type(error).__name__}: {error}'

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=5)
