#!/usr/bin/env python3
"""Check RViz URDF transforms against MuJoCo in zero and moving poses."""
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rotation(axis, angle):
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew @ skew)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['duck', 'character'], default='duck')
    args = parser.parse_args()
    model_file, urdf_file = ('bluefish_physics.xml', 'bluefish_physics.urdf') if args.variant == 'character' else ('chibi_fixed.xml', 'chibi_control.urdf')
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'mjcf' / model_file))
    data = mujoco.MjData(model)
    robot = ET.parse(ROOT / 'urdf' / urdf_file).getroot()
    joints = robot.findall('joint')
    for mesh in robot.findall('.//mesh'):
        path = mesh.get('filename').replace('package://chibi_sim/', '')
        assert (ROOT / path).is_file(), path
    rng = np.random.default_rng(42)
    for pose in range(11):
        values = {}
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            low, high = model.jnt_range[i]
            value = 0 if pose == 0 else rng.uniform(max(low, -.2), min(high, .2))
            data.qpos[model.jnt_qposadr[i]] = value
            values[name] = value
        mujoco.mj_forward(model, data)
        frames = {'base_link': np.eye(4)}
        for joint in joints:
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            origin = joint.find('origin')
            xyz = list(map(float, origin.get('xyz').split()))
            roll, pitch, yaw = map(float, origin.get('rpy').split())
            transform = np.eye(4)
            transform[:3, :3] = rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)
            transform[:3, 3] = xyz
            if joint.get('type') == 'revolute':
                turn = np.eye(4)
                turn[:3, :3] = rotation(list(map(float, joint.find('axis').get('xyz').split())), values[joint.get('name')])
                transform = transform @ turn
            frames[child] = frames[parent] @ transform
            if child.startswith('visual_'):
                continue
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, child.removeprefix('link_'))
            np.testing.assert_allclose(frames[child][:3, 3], data.xpos[body], atol=1e-9)
            np.testing.assert_allclose(frames[child][:3, :3], data.xmat[body].reshape(3, 3), atol=1e-9)
    if args.variant == 'character':
        assert not robot.findall('.//visual'), 'Custom appearance must not return'
        assert model.nmesh == 0, 'Physics skeleton must not contain custom meshes'
        print(f'PASS: physics-only skeleton; 11 poses, {len(joints)} transforms; appearance is downloaded VRM')
        return
    # Test placement as well as URDF/MJCF agreement: face must be on the
    # front of the head, and a frontal ray must reach each eye before the shell.
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'head_assembly')
    face = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'face_root')
    forward = data.xmat[face].reshape(3, 3)[:, 0].copy()
    for side in ('l', 'r'):
        eye = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'eye_' + side)
        eye_position = data.geom_xpos[eye].copy()
        assert abs(eye_position[2] - data.xpos[head, 2]) < .05, 'Eye is not at head height'
        assert eye_position[0] > data.xpos[head, 0] + .05, 'Eye is behind the front of the head'
        hit = np.array([-1], dtype=np.int32)
        mujoco.mj_ray(model, data, eye_position + forward*.5, -forward, None, True, -1, hit)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(hit[0])) if hit[0] >= 0 else None
        assert name in (f'eye_{side}', f'pupil_{side}', f'eye_highlight_{side}'), f'Eye {side} is occluded by {name}'
    print('PASS: both eyes are at head height and visible from the front')
    print(f'PASS: 11 poses, {len(joints)} link transforms, {len(robot.findall(".//visual"))} visuals, {model.njnt} moving joints')


if __name__ == '__main__':
    main()
