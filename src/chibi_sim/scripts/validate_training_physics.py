#!/usr/bin/env python3
"""Quantitative checks of the provisional training model, not hardware certification."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'mjcf' / 'bluefish_physics_free.xml'


def hull(points):
    points = sorted(set(map(tuple, points)))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    halves = []
    for seq in (points, points[::-1]):
        half = []
        for p in seq:
            while len(half) >= 2 and cross(half[-2], half[-1], p) <= 0:
                half.pop()
            half.append(p)
        halves.append(half[:-1])
    return halves[0] + halves[1]


def support_margin(model, data):
    points = []
    for side in ('left', 'right'):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, side+'_foot_contact')
        rotation = data.geom_xmat[gid].reshape(3, 3)
        size = model.geom_size[gid]
        for x in (-size[0], size[0]):
            for y in (-size[1], size[1]):
                points.append((data.geom_xpos[gid] + rotation @ [x, y, -size[2]])[:2])
    polygon, com = hull(points), data.subtree_com[1, :2]
    return float(min(((b[0]-a[0])*(com[1]-a[1]) - (b[1]-a[1])*(com[0]-a[0])) /
                     max(np.linalg.norm(np.array(b)-a), 1e-12)
                     for a, b in zip(polygon, polygon[1:]+polygon[:1])))


def simulate(path, seconds=4, dt=.002, mass_scale=1, friction=.8,
             drop=0, tilt_degrees=0):
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.timestep = dt
    model.body_mass[1:] *= mass_scale
    model.body_inertia[1:] *= mass_scale
    model.geom_friction[:, 0] = friction
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    mujoco.mj_resetData(model, data)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'root_free')
    qadr = model.jnt_qposadr[root]
    data.qpos[qadr+2] += drop
    angle = math.radians(tilt_degrees)
    data.qpos[qadr+3:qadr+7] = [math.cos(angle/2), 0, math.sin(angle/2), 0]
    mujoco.mj_forward(model, data)
    seen, loads, force_samples = set(), [], []
    max_velocity, max_penetration, max_energy = 0., 0., 0.
    for i in range(int(seconds/dt)):
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), 'nonfinite dynamics'
        max_velocity = max(max_velocity, float(np.max(np.abs(data.qvel))))
        max_energy = max(max_energy, float(np.sum(data.energy)))
        ground_force = 0.
        for c in data.contact[:data.ncon]:
            names = tuple(sorted(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                                 for g in (c.geom1, c.geom2)))
            seen.add(names)
            max_penetration = max(max_penetration, -float(c.dist))
        if i >= int(max(0, seconds-1)/dt):
            for cid in range(data.ncon):
                c = data.contact[cid]
                if 0 in (c.geom1, c.geom2):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, cid, force)
                    ground_force += force[0] * abs(c.frame[2])
            loads.append(ground_force)
    mujoco.mj_forward(model, data)
    z_axis = data.xmat[1].reshape(3, 3)[:, 2]
    tilt = math.degrees(math.acos(float(np.clip(z_axis[2], -1, 1))))
    weight = float(model.body_mass.sum() * abs(model.opt.gravity[2]))
    return {'root_xyz_m': data.qpos[qadr:qadr+3].tolist(),
            'com_xyz_m': data.subtree_com[1].tolist(), 'tilt_degrees': tilt,
            'support_margin_m': support_margin(model, data),
            'mass_kg': float(model.body_mass.sum()),
            'normal_load_n': float(np.mean(loads)), 'weight_n': weight,
            'load_error_ratio': abs(float(np.mean(loads))-weight)/weight,
            'max_penetration_m': max_penetration, 'max_state_velocity': max_velocity,
            'max_energy_j': max_energy, 'final_speed': float(np.linalg.norm(data.qvel)),
            'contacts': sorted(seen), 'warnings': [int(w.number) for w in data.warning]}


def validate(path, seconds=4):
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nq == 31 and model.nv == 30 and model.nu == 24
    assert model.nsensor == 5
    assert np.all(model.body_mass[1:] > 0), 'nonpositive body mass'
    inertia = model.body_inertia[1:]
    assert np.all(inertia > 0) and np.all(2*inertia.max(axis=1) <= inertia.sum(axis=1)+1e-12), 'invalid inertia'
    assert np.all(model.actuator_forcerange[:, 0] < model.actuator_forcerange[:, 1])
    results = {}
    cases = {'standing': {}, 'dt_1ms': {'dt': .001}, 'dt_4ms': {'dt': .004},
             'light_low_friction': {'mass_scale': .8, 'friction': .5},
             'heavy_high_friction': {'mass_scale': 1.2, 'friction': 1.0},
             'drop_10cm': {'drop': .1}}
    for name, kwargs in cases.items():
        result = simulate(path, seconds=seconds, **kwargs)
        assert result['tilt_degrees'] < 10, (name, 'not upright', result)
        assert result['final_speed'] < .1, (name, 'not settled', result)
        assert result['support_margin_m'] > .005, (name, 'COM outside support', result)
        assert result['load_error_ratio'] < .1, (name, 'contact force does not balance weight', result)
        assert result['max_energy_j'] > 0, 'energy computation disabled'
        assert not any(result['warnings']), (name, 'MuJoCo warnings', result)
        assert {('floor', 'left_foot_contact'), ('floor', 'right_foot_contact')}.issubset(set(map(tuple,result['contacts'])))
        results[name] = result
    nominal = np.array(results['standing']['root_xyz_m'])
    for name in ('dt_1ms', 'dt_4ms'):
        difference = float(np.linalg.norm(np.array(results[name]['root_xyz_m'])-nominal))
        assert difference < .005, (name, 'timestep-sensitive root position', difference)
        results[name]['root_difference_from_2ms_m'] = difference
    fall = simulate(path, seconds=seconds, drop=.12, tilt_degrees=65)
    assert any('floor' in pair and any(x in pair for x in ('head_contact', 'torso_contact'))
               for pair in fall['contacts']), ('missing whole-body fall collision',fall)
    results['fall_collision'] = fall
    return {'scope': 'provisional simulation model only; no hardware measurements',
            'status': 'passed', 'model': str(Path(path).resolve()), 'cases': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--seconds', type=float, default=4.)
    parser.add_argument('--steps', type=int, help='Compatibility: nominal 2 ms step count')
    parser.add_argument('--out', type=Path, default=ROOT.parents[1]/'validation/reports/bluefish_physics_final.json')
    args = parser.parse_args()
    report = validate(args.model, args.steps*.002 if args.steps else args.seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print('PASS:', len(report['cases']), 'cases;', args.out)
    for name, result in report['cases'].items():
        print(name, 'tilt_deg=', round(result['tilt_degrees'],4),
              'COM_margin_mm=',round(1000*result['support_margin_m'],2),
              'normal_N=',round(result['normal_load_n'],4))
