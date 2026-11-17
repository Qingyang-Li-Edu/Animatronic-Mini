#!/usr/bin/env python3
"""Summarize actuator effort and velocity from CSV or a ROS 2 bag."""
import argparse
import csv
import math
from pathlib import Path


def summarize(samples, limits=None):
    stats = {}
    for name, effort, velocity in samples:
        item = stats.setdefault(name, {'peak': 0.0, 'sq': 0.0, 'n': 0, 'speed': 0.0})
        item['peak'] = max(item['peak'], abs(effort))
        item['sq'] += effort * effort
        item['n'] += 1
        item['speed'] = max(item['speed'], abs(velocity))
    result = {}
    for name, v in stats.items():
        if not v['n']:
            continue
        item = {'peak_torque': round(v['peak'], 4),
                'rms_torque': round(math.sqrt(v['sq'] / v['n']), 4),
                'max_velocity': round(v['speed'], 4)}
        if limits and name in limits:
            threshold = 0.7 * limits[name]
            item['peak_percent_limit'] = round(100 * v['peak'] / limits[name], 1)
            item['under_70_percent'] = v['peak'] < threshold
        result[name] = item
    return result


def model_torque_limits(path):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(path))
    limits = {}
    for actuator in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator, 0])
        joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        limits[joint] = max(abs(float(v)) for v in model.actuator_forcerange[actuator])
    return limits


def read_csv(path):
    samples = []
    with open(path, newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            for key, value in row.items():
                if key.endswith('_effort') and value:
                    joint = key[:-7]
                    samples.append((joint, float(value), float(row.get(joint + '_velocity') or 0)))
    return samples


def read_bag(path, topic):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        raise RuntimeError('Read rosbag input inside a sourced ROS 2 environment') from exc
    reader = rosbag2_py.SequentialReader()
    storage_id = 'mcap' if any(Path(path).glob('*.mcap')) else 'sqlite3'
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=storage_id),
                rosbag2_py.ConverterOptions('', ''))
    samples = []
    while reader.has_next():
        name, payload, _ = reader.read_next()
        if name != topic:
            continue
        message = deserialize_message(payload, JointState)
        for i, joint in enumerate(message.name):
            if i < len(message.effort):
                velocity = message.velocity[i] if i < len(message.velocity) else 0.0
                samples.append((joint, float(message.effort[i]), float(velocity)))
    return samples


def plot(samples, output):
    import matplotlib.pyplot as plt
    joints = sorted({row[0] for row in samples})
    fig, axes = plt.subplots(max(1, len(joints)), 1, figsize=(8, max(3, 2.4 * len(joints))), squeeze=False)
    for ax, joint in zip(axes[:, 0], joints):
        rows = [r for r in samples if r[0] == joint]
        ax.scatter([r[2] for r in rows], [r[1] for r in rows], s=8, alpha=.55)
        ax.set(title=joint, xlabel='Angular velocity (rad/s)', ylabel='Effort (N m)')
        ax.grid(True, alpha=.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?', help='JointState CSV or rosbag2 directory')
    parser.add_argument('--topic', default='/mujoco/joint_states')
    parser.add_argument('--plot', default='effort_velocity.png')
    parser.add_argument('--model', default=str(Path(__file__).resolve().parents[1] / 'mjcf/chibi_fixed.xml'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(summarize([('neck', 1, 2), ('neck', -2, 1)]))
        return
    if not args.input:
        parser.error('input CSV or rosbag2 directory required')
    source = Path(args.input)
    samples = read_csv(source) if source.is_file() else read_bag(source, args.topic)
    if not samples:
        raise RuntimeError(f'No effort samples found in {source}')
    limits = model_torque_limits(args.model)
    for joint, values in summarize(samples, limits).items():
        status = 'OK' if values.get('under_70_percent', False) else 'CHECK' if 'under_70_percent' in values else 'N/A'
        ratio = f", {values['peak_percent_limit']}% of configured limit" if 'peak_percent_limit' in values else ''
        print(f"{joint}: peak={values['peak_torque']} N m, RMS={values['rms_torque']} N m, max speed={values['max_velocity']} rad/s{ratio} [{status}]")
    plot(samples, args.plot)
    print(f'Scatter plot: {args.plot}')


if __name__ == '__main__':
    main()
