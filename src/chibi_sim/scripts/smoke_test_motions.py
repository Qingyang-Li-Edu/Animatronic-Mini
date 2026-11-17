#!/usr/bin/env python3
"""Publish the motion library and verify the MuJoCo state stream."""
import time
import os
from pathlib import Path
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

MOTIONS = ['nod', 'shake', 'look_around', 'crouch', 'greet', 'blink']
CHARACTER = os.environ.get('CHIBI_VARIANT') == 'character'
EXPECTED_JOINTS = int(os.environ.get('CHIBI_EXPECTED_JOINTS', '24' if CHARACTER else '20'))


class MotionSmokeTest(Node):
    def __init__(self):
        super().__init__('chibi_motion_smoke_test')
        self.publisher = self.create_publisher(String, '/chibi/motion', 10)
        self.samples = 0
        self.latest = None
        self.position_ranges = {}
        self.max_velocity = {}
        self.create_subscription(JointState, '/mujoco/joint_states', self.on_state, 10)

    def on_state(self, message):
        self.samples += 1
        self.latest = message
        for name, value in zip(message.name, message.position):
            low, high = self.position_ranges.get(name, (float(value), float(value)))
            self.position_ranges[name] = (min(low, value), max(high, value))
        for name, value in zip(message.name, message.velocity):
            self.max_velocity[name] = max(self.max_velocity.get(name, 0.0), abs(value))


def main():
    rclpy.init()
    node = MotionSmokeTest()
    motion_name = 'motions_character.yaml' if CHARACTER else 'motions.yaml'
    motion_file = Path(os.environ.get('CHIBI_MOTIONS', Path(__file__).resolve().parents[1] / 'config' / motion_name))
    motions = yaml.safe_load(motion_file.read_text())['motions']
    try:
        deadline = time.monotonic() + 1.0
        for motion in MOTIONS:
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=min(0.1, deadline - time.monotonic()))
            node.publisher.publish(String(data=motion))
            print(f'triggered {motion}', flush=True)
            duration = max((float(point[0]) for track in motions[motion].values() for point in track.get('points', [])), default=0)
            deadline = time.monotonic() + duration + 0.6
        end = deadline
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.samples == 0 or node.latest is None:
            raise RuntimeError('No /mujoco/joint_states messages received')
        if len(node.latest.name) != EXPECTED_JOINTS:
            raise RuntimeError(f'Expected {EXPECTED_JOINTS} joints, got {len(node.latest.name)}')
        if not (len(node.latest.position) == len(node.latest.velocity) == len(node.latest.effort) == EXPECTED_JOINTS):
            raise RuntimeError('JointState position, velocity, and effort arrays must contain all joints')
        body_motion = max(node.position_ranges['head_pitch'][1] - node.position_ranges['head_pitch'][0],
                          node.position_ranges['head_yaw'][1] - node.position_ranges['head_yaw'][0])
        face_motion = node.position_ranges['eyelid_close'][1] - node.position_ranges['eyelid_close'][0]
        if body_motion < 0.05 or face_motion < 0.1:
            raise RuntimeError(f'Expected body and face motion, measured ranges {body_motion:.3f}, {face_motion:.3f}')
        if max(node.max_velocity.values(), default=0.0) < 0.05:
            raise RuntimeError('No non-zero joint velocity observed')
        if CHARACTER:
            arm_range = node.position_ranges['left_shoulder'][1] - node.position_ranges['left_shoulder'][0]
            if arm_range < 0.2:
                raise RuntimeError(f'No greeting arm movement: range={arm_range:.3f}')
        print(f'PASS: {len(MOTIONS)} motions, {node.samples} state samples, {EXPECTED_JOINTS} joints')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
