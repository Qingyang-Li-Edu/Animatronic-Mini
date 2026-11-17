"""Free-root standing under finite horizontal pushes, using provisional physics."""
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / 'src/chibi_sim/mjcf/bluefish_physics_free.xml'


class CharacterBalanceEnv(gym.Env):
    metadata = {'render_modes': []}
    version = 1

    def __init__(self, push_force=.8, randomize=True, episode_seconds=6.,
                 force_cap_ratio=.65, action_scale=.12):
        if push_force < 0 or episode_seconds <= 0 or not 0 < force_cap_ratio <= 1:
            raise ValueError('Invalid force, duration, or force cap')
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 5
        self.dt = self.model.opt.timestep * self.frame_skip
        self.episode_steps = round(episode_seconds / self.dt)
        self.push_force, self.randomize = float(push_force), bool(randomize)
        self.action_scale = float(action_scale)
        self.base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base')
        self.floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.feet = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, s+'_foot_contact')
                     for s in ('left', 'right')]
        self.jids = self.model.actuator_trnid[:, 0].copy()
        self.qadr = self.model.jnt_qposadr[self.jids]
        self.vadr = self.model.jnt_dofadr[self.jids]
        self.controlled = np.asarray([i for i, j in enumerate(self.jids)
            if any(k in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(j))
                   for k in ('hip_', 'knee', 'ankle'))])
        self.mass = self.model.body_mass.copy()
        self.inertia = self.model.body_inertia.copy()
        self.friction = self.model.geom_friction.copy()
        self.original_limits = np.max(np.abs(self.model.actuator_forcerange), axis=1).copy()
        self.model.actuator_forcerange *= force_cap_ratio
        self.action_space = spaces.Box(-1., 1., shape=(len(self.controlled),), dtype=np.float32)
        # projected gravity, local base velocity, joint state, previous action,
        # two foot contact flags, root height. No access to future disturbances.
        self.observation_space = spaces.Box(-np.inf, np.inf,
            shape=(3+6+2*self.model.nu+len(self.controlled)+2+1,), dtype=np.float32)
        self.previous_action = np.zeros(len(self.controlled), dtype=np.float32)

    def _obs(self):
        rotation = self.data.xmat[self.base].reshape(3, 3)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.base, vel, 1)
        touches = np.zeros(2)
        for c in self.data.contact[:self.data.ncon]:
            if self.floor in (c.geom1, c.geom2):
                for i, foot in enumerate(self.feet):
                    if foot in (c.geom1, c.geom2):
                        touches[i] = 1.
        return np.concatenate((rotation.T @ [0., 0., -1.], np.clip(vel, -20, 20),
            self.data.qpos[self.qadr], np.clip(self.data.qvel[self.vadr], -20, 20),
            self.previous_action, touches, [self.data.qpos[2]])).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        self.model.body_mass[:] = self.mass
        self.model.body_inertia[:] = self.inertia
        self.model.geom_friction[:] = self.friction
        if self.randomize:
            scales = rng.uniform(.8, 1.2, self.model.nbody-1)
            self.model.body_mass[1:] *= scales
            self.model.body_inertia[1:] *= scales[:, None]
            self.model.geom_friction[:, 0] = rng.uniform(.5, 1.)
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        if self.randomize:
            axis = np.array([*rng.normal(size=2), 0.])
            axis /= max(np.linalg.norm(axis), 1e-10)
            mujoco.mju_axisAngle2Quat(self.data.qpos[3:7], axis, rng.uniform(-.035, .035))
            self.data.qpos[self.qadr] = np.clip(rng.normal(0, .005, self.model.nu),
                self.model.jnt_range[self.jids, 0], self.model.jnt_range[self.jids, 1])
            self.data.qvel[:] = rng.normal(0, .005, self.model.nv)
        # Keep push direction and force outside observations; the controller
        # must react to measured dynamics rather than an oracle push signal.
        angle = float(options.get('push_angle', rng.uniform(-np.pi, np.pi)))
        force = float(options.get('push_force', self.push_force))
        self.push_vector = force * np.array([np.cos(angle), np.sin(angle), 0.])
        self.push_times = (1., 2.5, 4.)
        self.push_duration = .12
        self.step_count = 0
        self.previous_action[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.start_xy = self.data.qpos[:2].copy()
        return self._obs(), {'push_force_n': force, 'push_angle_rad': angle}

    def _set_control(self, action):
        """Separate command composition from the unchanged physics step."""
        self.data.ctrl[:] = 0
        self.data.ctrl[self.controlled] = self.action_scale * action
        self.data.ctrl[:] = np.clip(self.data.ctrl, self.model.actuator_ctrlrange[:, 0],
                                    self.model.actuator_ctrlrange[:, 1])

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError('Expected finite balance action vector')
        action = action.clip(-1, 1)
        self._set_control(action)
        force_peak, penetration, nonfoot_contact = 0., 0., False
        for _ in range(self.frame_skip):
            self.data.xfrc_applied[:] = 0
            if any(t <= self.data.time < t+self.push_duration for t in self.push_times):
                self.data.xfrc_applied[self.base, :3] = self.push_vector
            mujoco.mj_step(self.model, self.data)
            force_peak = max(force_peak,
                float(np.max(np.abs(self.data.actuator_force)/self.original_limits)))
            for c in self.data.contact[:self.data.ncon]:
                penetration = max(penetration, -float(c.dist))
                if self.floor in (c.geom1, c.geom2):
                    other = c.geom2 if c.geom1 == self.floor else c.geom1
                    nonfoot_contact |= other not in self.feet
        self.step_count += 1
        mujoco.mj_forward(self.model, self.data)
        upright = float(self.data.xmat[self.base].reshape(3, 3)[2, 2])
        tilt = float(np.arccos(np.clip(upright, -1, 1)))
        speed = float(np.linalg.norm(self.data.qvel[:6]))
        drift = float(np.linalg.norm(self.data.qpos[:2]-self.start_xy))
        finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        warning = any(w.number for w in self.data.warning)
        fallen = tilt > np.deg2rad(35) or self.data.qpos[2] < .115 or nonfoot_contact
        terminated = not finite or warning or fallen
        # Alive and upright, minimal base motion and drift, gentle joint inputs.
        cost = (80*tilt**2 + 2*np.sum(self.data.qvel[:3]**2) +
                .05*np.sum(self.data.qvel[3:6]**2) + 100*drift**2 +
                .03*np.mean(action**2) + .03*np.mean((action-self.previous_action)**2))
        reward = 1.-float(cost) if finite else -100.
        if terminated:
            reward -= 20.
        self.previous_action[:] = action
        info = {'tilt_deg': float(np.rad2deg(tilt)), 'base_speed': speed,
                'drift_m': drift, 'height_m': float(self.data.qpos[2]),
                'max_force_ratio': force_peak, 'max_penetration_m': penetration,
                'nonfoot_contact': bool(nonfoot_contact), 'fallen': bool(fallen),
                'warnings': bool(warning)}
        return self._obs(), reward, bool(terminated), self.step_count >= self.episode_steps, info
