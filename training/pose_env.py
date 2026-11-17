#!/usr/bin/env python3
"""Fixed-base pose tracking environment for the character action library."""
from pathlib import Path
import numpy as np
import yaml
import mujoco

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # Keep model inspection usable before optional RL installs.
    gym = None
    spaces = None

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'src' / 'chibi_sim' / 'mjcf' / 'bluefish_physics.xml'
MOTIONS_PATH = ROOT / 'src' / 'chibi_sim' / 'config' / 'motions_character.yaml'


class CharacterPoseEnv(gym.Env if gym else object):
    """Learn direct actuator commands that track one motion trajectory."""
    metadata = {'render_modes': []}

    def __init__(self, model_path=MODEL_PATH, motions_path=MOTIONS_PATH,
                 motion='greet', frame_skip=5, episode_seconds=3.0,
                 action_mode='absolute', residual_scale=.12, force_cap_ratio=1.):
        if spaces is None:
            raise RuntimeError('Install gymnasium before creating the training environment')
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = int(frame_skip)
        if self.frame_skip < 1 or episode_seconds <= 0:
            raise ValueError('Positive frame_skip and episode_seconds required')
        self.episode_steps = max(1, int(episode_seconds / (self.model.opt.timestep * self.frame_skip)))
        self.motion = motion
        if action_mode not in ('absolute', 'residual'):
            raise ValueError('Unknown action mode')
        self.action_mode = action_mode
        self.residual_scale = float(residual_scale)
        if not 0 < force_cap_ratio <= 1:
            raise ValueError('force_cap_ratio must be in (0, 1]')
        self.configured_force_limits = np.max(np.abs(self.model.actuator_forcerange),axis=1).copy()
        self.model.actuator_forcerange *= force_cap_ratio
        self.motion_data = yaml.safe_load(Path(motions_path).read_text(encoding='utf-8'))['motions']
        self.set_motion(motion)
        self.joints = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                       for i in range(self.model.njnt)]
        self.qadr = {name: int(self.model.jnt_qposadr[i]) for i, name in enumerate(self.joints)}
        self.dadr = {name: int(self.model.jnt_dofadr[i]) for i, name in enumerate(self.joints)}
        self.actuator_joints = []
        for actuator in range(self.model.nu):
            jid = int(self.model.actuator_trnid[actuator, 0])
            self.actuator_joints.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid))
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_size = len(self.joints) * 3 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.step_count = 0
        self.previous_action = np.zeros(self.model.nu,dtype=np.float32)
        self.np_random = np.random.default_rng()
        self.active_indices = [i for i, name in enumerate(self.actuator_joints)
                               if any(name in track.get('joint_names', [])
                                      for track in self.motion_data[self.motion].values())]

    def set_motion(self, name):
        if name not in self.motion_data:
            raise ValueError(f'unknown motion: {name}')
        self.motion = name

    def _duration(self):
        return max((float(row[0]) for track in self.motion_data[self.motion].values()
                    for row in track.get('points', [])), default=1.0)

    def _target(self, seconds):
        target = {name: 0.0 for name in self.joints}
        for track in self.motion_data[self.motion].values():
            names, rows = track.get('joint_names', []), track.get('points', [])
            if not names or not rows:
                continue
            if seconds <= float(rows[0][0]):
                values = rows[0][1:]
            elif seconds >= float(rows[-1][0]):
                values = rows[-1][1:]
            else:
                for left, right in zip(rows, rows[1:]):
                    if float(left[0]) <= seconds <= float(right[0]):
                        alpha = ((seconds - float(left[0])) /
                                 max(1e-9, float(right[0]) - float(left[0])))
                        values = [float(a) + alpha * (float(b) - float(a))
                                  for a, b in zip(left[1:], right[1:])]
                        break
            target.update(zip(names, map(float, values)))
        return np.asarray([target[name] for name in self.actuator_joints], dtype=np.float32)

    def _obs(self, target):
        q = np.asarray([self.data.qpos[self.qadr[name]] for name in self.joints], dtype=np.float32)
        v = np.asarray([self.data.qvel[self.dadr[name]] for name in self.joints], dtype=np.float32)
        return np.concatenate((q, np.clip(v, -20, 20), target, [self.step_count / self.episode_steps])).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and options.get('motion'):
            self.set_motion(options['motion'])
            self.active_indices = [i for i, name in enumerate(self.actuator_joints)
                                   if any(name in track.get('joint_names', [])
                                          for track in self.motion_data[self.motion].values())]
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.previous_action[:] = 0
        mujoco.mj_forward(self.model, self.data)
        target = self._target(0.0)
        return self._obs(target), {'motion': self.motion}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.model.nu,) or not np.isfinite(action).all():
            raise ValueError('Expected a finite action vector matching model.nu')
        action = action.clip(-1, 1)
        if self.action_mode == 'residual':
            seconds = self.step_count * self.frame_skip * self.model.opt.timestep
            control = self._target(seconds) + self.residual_scale * action
            control = np.clip(control, self.ctrl_low, self.ctrl_high)
        else:
            control = self.ctrl_low + (action + 1) * .5 * (self.ctrl_high - self.ctrl_low)
        self.data.ctrl[:] = control
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        seconds = self.step_count * self.frame_skip * self.model.opt.timestep
        target = self._target(seconds)
        q = np.asarray([self.data.qpos[self.qadr[name]] for name in self.actuator_joints])
        v = np.asarray([self.data.qvel[self.dadr[name]] for name in self.actuator_joints])
        error = q - target
        force_ratio = self.data.actuator_force/self.configured_force_limits
        reward = -float(np.mean(error * error) * 1000.0 + np.mean(v * v) * .001 +
                        np.mean(action * action) * .002 + .1*np.mean(force_ratio**2) +
                        .01*np.mean((action-self.previous_action)**2))
        self.previous_action[:] = action
        terminated = not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel))
        truncated = self.step_count >= self.episode_steps
        return self._obs(target), reward, bool(terminated), bool(truncated), {
            'motion': self.motion, 'q_error_rms': float(np.sqrt(np.mean(error * error))),
            'active_q_error_rms': float(np.sqrt(np.mean(error[self.active_indices] ** 2)))}
