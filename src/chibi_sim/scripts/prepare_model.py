#!/usr/bin/env python3
"""Generate fixed-base chibi MJCF from the vendored Open Duck Mini v2 model."""
from pathlib import Path
import math
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'third_party/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/open_duck_mini_v2.xml'
OUTPUT_DIR = ROOT / 'src/chibi_sim/mjcf'
ASSETS = SOURCE.parent / 'assets'
URDF_OUTPUT = ROOT / 'src/chibi_sim/urdf/chibi_control.urdf'


def add_body_inertial(body):
    ET.SubElement(body, 'inertial', pos='0 0 0', mass='0.00001', diaginertia='0.000001 0.000001 0.000001')


def make_hinge_body(parent, name, joint_name, pos, axis, limits):
    body = ET.SubElement(parent, 'body', name=name, pos=pos)
    add_body_inertial(body)
    ET.SubElement(body, 'joint', name=joint_name, type='hinge', axis=axis, range=limits,
                   damping='0.03', armature='0.0001')
    return body


def add_face(head):
    # head_assembly's local X points up and local -Z points forward.
    # Rotate the face frame so +X faces forward and +Z points up.
    root = ET.SubElement(head, 'body', name='face_root', pos='0.010 0 -0.065',
                         quat='0.7071067811865476 0 0.7071067811865475 0')
    add_body_inertial(root)
    ET.SubElement(root, 'geom', name='face_panel', type='box', pos='0 0 0',
                  size='0.003 0.085 0.025', rgba='0.025 0.055 0.065 1',
                  contype='0', conaffinity='0')
    pan = make_hinge_body(root, 'face_pan', 'eye_pan', '0 0 0', '0 0 1', '-0.5 0.5')
    tilt = make_hinge_body(pan, 'face_tilt', 'eye_tilt', '0 0 0', '0 1 0', '-0.35 0.35')
    for side, y in (('l', 0.040), ('r', -0.040)):
        ET.SubElement(tilt, 'geom', name=f'eye_{side}', type='sphere',
                      pos=f'0.015 {y} 0.009', size='0.013', rgba='0.80 0.96 1 1',
                      contype='0', conaffinity='0')
        ET.SubElement(tilt, 'geom', name=f'pupil_{side}', type='sphere',
                      pos=f'0.026 {y} 0.009', size='0.0075', rgba='0.015 0.025 0.035 1',
                      contype='0', conaffinity='0')
        ET.SubElement(tilt, 'geom', name=f'eye_highlight_{side}', type='sphere',
                      pos=f'0.032 {y+0.002} 0.012', size='0.002', rgba='1 1 1 1',
                      contype='0', conaffinity='0')
    eyelid = make_hinge_body(root, 'face_eyelid', 'eyelid_close', '0 0 0.030', '0 1 0', '0 1.2')
    ET.SubElement(eyelid, 'geom', name='eyelid', type='box', pos='0.027 0 0',
                  size='0.004 0.061 0.004', rgba='0.10 0.45 0.50 1', contype='0', conaffinity='0')
    jaw = make_hinge_body(root, 'face_jaw', 'jaw_open', '0 0 -0.014', '0 1 0', '0 0.8')
    ET.SubElement(jaw, 'geom', name='jaw', type='box', pos='0.008 0 -0.004',
                  size='0.004 0.025 0.003', rgba='0.55 0.87 0.91 1', contype='0', conaffinity='0')


def articulate_antennas(head, root):
    source_geoms = [g for g in list(head) if g.tag == 'geom' and g.get('mesh') in
                    {'antenna', 'left_antenna_holder', 'right_antenna_holder'}]
    buckets = {'antenna_l': [], 'antenna_r': []}
    for geom in source_geoms:
        mesh = geom.get('mesh')
        if mesh == 'left_antenna_holder':
            buckets['antenna_l'].append(geom)
        elif mesh == 'right_antenna_holder':
            buckets['antenna_r'].append(geom)
        elif mesh == 'antenna':
            y = float(geom.get('pos', '0 0 0').split()[1])
            buckets['antenna_l' if y > 0.05 else 'antenna_r'].append(geom)
    for geom in source_geoms:
        head.remove(geom)
    for name, geoms in buckets.items():
        pivot = [-0.12906, 0.0, 0.06095]
        body = make_hinge_body(head, name, name, ' '.join(map(str, pivot)), '1 0 0', '-0.8 0.8')
        for geom in geoms:
            pos = [float(v) for v in geom.get('pos', '0 0 0').split()]
            geom.set('pos', ' '.join(f'{v-p:.9g}' for v, p in zip(pos, pivot)))
            body.append(geom)


def write_urdf(model_root, output=None, robot_name='chibi', split_visuals=False):
    output = Path(output) if output is not None else URDF_OUTPUT
    robot = ET.Element('robot', name=robot_name)
    joints = [j for j in model_root.findall('.//joint') if j.get('name')]
    actuator_forces = {}
    for actuator in model_root.findall('./actuator/position'):
        bounds = actuator.get('forcerange')
        if bounds:
            actuator_forces[actuator.get('joint')] = max(abs(float(x)) for x in bounds.split())
        elif actuator.get('class') == 'sts3215':
            actuator_forces[actuator.get('joint')] = 3.23
    # Preserve the MJCF body tree and mesh frames for RViz TF rendering.
    def origin(parent, element):
        w, x, y, z = map(float, element.get('quat', '1 0 0 0').split())
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = (v / norm for v in (w, x, y, z))
        r00, r10, r20 = 1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)
        r11, r12 = 1-2*(x*x+z*z), 2*(y*z-w*x)
        r21, r22 = 2*(y*z+w*x), 1-2*(x*x+y*y)
        horizontal = math.hypot(r00, r10)
        pitch = math.atan2(-r20, horizontal)
        if horizontal < 1e-7:
            rpy = (math.atan2(-r12, r11), pitch, 0.0)
        else:
            rpy = (math.atan2(r21, r22), pitch, math.atan2(r10, r00))
        ET.SubElement(parent, 'origin', xyz=element.get('pos', '0 0 0'),
                      rpy=' '.join(f'{v:.12g}' for v in rpy))

    meshes = {m.get('name', Path(m.get('file')).stem): m for m in model_root.findall('asset/mesh')}
    materials = {m.get('name'): m.get('rgba', '0.8 0.8 0.8 1')
                 for m in model_root.findall('asset/material')}
    ET.SubElement(robot, 'link', name='base_link')

    def body_link(body, parent):
        name = 'link_' + body.get('name')
        link = ET.SubElement(robot, 'link', name=name)
        movable = body.findall('joint')
        if len(movable) > 1:
            raise ValueError(f'Multiple joints in body {name} require intermediate links')
        source = movable[0] if movable else None
        if source is not None and source.get('pos', '0 0 0') != '0 0 0':
            raise ValueError(f'Unsupported nonzero joint anchor in {name}')
        joint_name = source.get('name') if source is not None else name + '_fixed'
        item = ET.SubElement(robot, 'joint', name=joint_name,
                             type='revolute' if source is not None else 'fixed')
        ET.SubElement(item, 'parent', link=parent)
        ET.SubElement(item, 'child', link=name)
        origin(item, body)
        if source is not None:
            ET.SubElement(item, 'axis', xyz=source.get('axis', '0 0 1'))
            low, high = source.get('range', '-3.141593 3.141593').split()
            ET.SubElement(item, 'limit', lower=low, upper=high,
                          effort=str(actuator_forces.get(joint_name, 0.08)), velocity='8.0')
        for index, geom in enumerate(body.findall('geom')):
            kind = geom.get('type', 'mesh' if geom.get('mesh') else '')
            if geom.get('class') == 'collision' or kind not in ('mesh', 'sphere', 'box'):
                continue
            visual_parent = link
            if split_visuals:
                visual_name = 'visual_' + geom.get('name', f'{name}_{index}')
                visual_parent = ET.SubElement(robot, 'link', name=visual_name)
                fixed = ET.SubElement(robot, 'joint', name=visual_name + '_fixed', type='fixed')
                ET.SubElement(fixed, 'parent', link=name)
                ET.SubElement(fixed, 'child', link=visual_name)
                origin(fixed, geom)
            visual = ET.SubElement(visual_parent, 'visual')
            if not split_visuals:
                origin(visual, geom)
            geometry = ET.SubElement(visual, 'geometry')
            if kind == 'mesh':
                mesh = meshes[geom.get('mesh')]
                ET.SubElement(geometry, 'mesh',
                              filename='package://chibi_sim/mjcf/assets/' + mesh.get('file'),
                              scale=mesh.get('scale', '1 1 1'))
            elif kind == 'sphere':
                ET.SubElement(geometry, 'sphere', radius=geom.get('size').split()[0])
            else:
                ET.SubElement(geometry, 'box', size=' '.join(str(2*float(v)) for v in geom.get('size').split()))
            material = ET.SubElement(visual, 'material', name=f'{name}_{index}')
            ET.SubElement(material, 'color', rgba=geom.get('rgba', materials.get(geom.get('material'), '0.8 0.8 0.8 1')))
        for child in body.findall('body'):
            body_link(child, name)

    for body in model_root.findall('worldbody/body'):
        body_link(body, 'base_link')
    control = ET.SubElement(robot, 'ros2_control', name='ChibiTopicSystem', type='system')
    hardware = ET.SubElement(control, 'hardware')
    ET.SubElement(hardware, 'plugin').text = 'joint_state_topic_hardware_interface/JointStateTopicSystem'
    ET.SubElement(hardware, 'param', name='joint_commands_topic').text = '/mujoco/joint_commands'
    ET.SubElement(hardware, 'param', name='joint_states_topic').text = '/mujoco/joint_states'
    ET.SubElement(hardware, 'param', name='trigger_joint_command_threshold').text = '-1'
    for joint in joints:
        item = ET.SubElement(control, 'joint', name=joint.get('name'))
        ET.SubElement(item, 'command_interface', name='position')
        for interface in ('position', 'velocity', 'effort'):
            state = ET.SubElement(item, 'state_interface', name=interface)
            if interface == 'position':
                ET.SubElement(state, 'param', name='initial_value').text = '0'
    ET.indent(robot, space='  ')
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(output, encoding='utf-8', xml_declaration=True)


def main():
    if not SOURCE.is_file():
        raise SystemExit(f'Missing source model: {SOURCE}')
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set('model', 'chibi_fixed')
    root.find('compiler').set('meshdir', 'assets')
    option = root.find('option')
    option.set('timestep', '0.002')
    base = root.find("worldbody/body[@name='base']")
    base.remove(base.find('freejoint'))
    head = root.find(".//body[@name='head_assembly']")
    if head is None:
        raise SystemExit('Could not find head_assembly in source MJCF')
    articulate_antennas(head, root)
    add_face(head)
    floor = ET.Element('geom', name='floor', type='plane', pos='0 0 0', size='0 0 0.02',
                       contype='1', conaffinity='1', friction='0.8 0.1 0.1', condim='3',
                       rgba='0.86 0.88 0.9 1')
    root.find('worldbody').append(floor)
    actuator = root.find('actuator')
    for name, force, kp, limits in (
        ('antenna_l', 0.3, 8, '-0.8 0.8'), ('antenna_r', 0.3, 8, '-0.8 0.8'),
        ('eye_pan', 0.08, 4, '-0.5 0.5'), ('eye_tilt', 0.08, 4, '-0.35 0.35'),
        ('eyelid_close', 0.08, 4, '0 1.2'), ('jaw_open', 0.08, 4, '0 0.8')):
        ET.SubElement(actuator, 'position', name=f'act_{name}', joint=name, kp=str(kp),
                      ctrlrange=limits, forcerange=f'{-force} {force}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ASSETS, OUTPUT_DIR / 'assets', dirs_exist_ok=True)
    ET.indent(tree, space='  ')
    tree.write(OUTPUT_DIR / 'chibi_fixed.xml', encoding='utf-8', xml_declaration=True)
    write_urdf(root)
    print(f'Generated {OUTPUT_DIR / "chibi_fixed.xml"}')
    print(f'Generated {URDF_OUTPUT}')


if __name__ == '__main__':
    main()
