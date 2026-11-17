"""Free-root greeting: pose PPO controls the upper body, balance controls legs."""
from pathlib import Path
import numpy as np
import mujoco
from stable_baselines3 import PPO

from balance_env import CharacterBalanceEnv
from pose_env import CharacterPoseEnv

ROOT = Path(__file__).resolve().parents[1]
POSE_POLICY = ROOT/'training/checkpoints/ppo-safe/greet.zip'


class GreetBalanceEnv(CharacterBalanceEnv):
    """Retain the 70D balance input to reuse and fine-tune the standing policy.

    This is a hierarchical controller, not a new end-to-end full-body policy.
    Both controllers use current measured state; the upper body reference
    repeats every 3 seconds after an initial half-second settling period.
    """
    version = 1

    def __init__(self, upper_controller='ppo', episode_seconds=8., **kwargs):
        if episode_seconds < 4.5:
            raise ValueError('Greeting requires settling, full cycles and a quiet tail')
        super().__init__(episode_seconds=episode_seconds, **kwargs)
        if upper_controller not in ('ppo', 'reference'):
            raise ValueError('Unknown upper controller')
        self.upper_controller = upper_controller
        self.pose = CharacterPoseEnv(motion='greet', action_mode='residual', force_cap_ratio=.65)
        self.pose_policy = PPO.load(POSE_POLICY, device='cpu') if upper_controller == 'ppo' else None
        self.upper = np.asarray([i for i in range(self.model.nu) if i not in self.controlled])
        self.arm_indices = np.asarray([i for i, n in enumerate(self.pose.actuator_joints)
                                      if 'shoulder' in n or 'elbow' in n])
        self.greet_target = np.zeros(self.model.nu, dtype=np.float32)
        self.motion_end = .5 + 3*int((episode_seconds-1.5)//3)
        self.upper_enabled = True

    def set_upper_enabled(self, enabled):
        self.upper_enabled = bool(enabled)

    def _phase(self):
        if self.data.time < .5:
            return 0.
        if self.data.time >= self.motion_end:
            return 3.
        return (self.data.time-.5) % 3.

    def _set_control(self, action):
        super()._set_control(action)
        phase = self._phase()
        self.greet_target = self.pose._target(phase)
        if self.data.time < .5 or not self.upper_enabled:
            return
        self.pose.data.qpos[:] = self.data.qpos[self.qadr]
        self.pose.data.qvel[:] = self.data.qvel[self.vadr]
        self.pose.step_count = min(int(phase/self.dt), self.pose.episode_steps-1)
        pose_action = (self.pose_policy.predict(self.pose._obs(self.greet_target), deterministic=True)[0]
                       if self.pose_policy else np.zeros(self.model.nu))
        commands = self.greet_target + self.pose.residual_scale*pose_action
        self.data.ctrl[self.upper] = np.clip(commands[self.upper],
            self.model.actuator_ctrlrange[self.upper, 0], self.model.actuator_ctrlrange[self.upper, 1])

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        target = self.pose._target(self._phase())
        error = self.data.qpos[self.qadr[self.arm_indices]]-target[self.arm_indices]
        # Balance is still primary; report arm tracking separately.
        info['arm_rmse_rad'] = float(np.sqrt(np.mean(error**2)))
        info['greet_phase_s'] = self._phase()
        info['upper_controller'] = self.upper_controller
        info['settling'] = bool(self.data.time >= self.motion_end)
        # Stronger damping and position retention for moving-arm compensation.
        reward -= float(.2*np.sum(self.data.qvel[3:6]**2) + 300*info['drift_m']**2)
        return obs, reward, term, trunc, info

    def close(self):
        self.pose.close()
        super().close()
