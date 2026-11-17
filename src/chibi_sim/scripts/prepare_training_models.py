#!/usr/bin/env python3
"""Create training variants from the visual character MJCF.

The generated free model keeps the visual meshes and actuators, adds a free
root joint, low complexity foot collision geometry, and state sensors. It is a
training model, not a hardware model.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'mjcf' / 'bluefish_physics.xml'
OUTPUT = ROOT / 'mjcf' / 'bluefish_physics_free.xml'


def child(parent, tag, **attrs):
    return ET.SubElement(parent, tag, **attrs)


def make_free_model(source=SOURCE, output=OUTPUT):
    root = ET.parse(source).getroot()
    root.set('model', 'bluefish_physics_free')
    option = root.find('option')
    option.set('iterations', '50')
    option.set('timestep', '0.002')
    flag = option.find('flag')
    if flag is None:
        flag = child(option, 'flag')
    flag.set('energy', 'enable')
    world = root.find('worldbody')
    base = world.find("body[@name='base']")
    if base is None:
        raise ValueError('base body not found')
    if base.find("freejoint[@name='root_free']") is None:
        base.insert(0, ET.Element('freejoint', {'name': 'root_free'}))

    # Whole-body collision proxies catch falls and non-adjacent self-contact.
    # They deliberately do not use the copyrighted visual avatar's meshes.
    common = dict(rgba='0 0 0 0', contype='1', conaffinity='1',
                  friction='0.8 0.03 0.01', solref='0.015 1',
                  solimp='0.90 0.95 0.02', group='3', mass='0')
    child(base, 'geom', name='torso_contact', type='ellipsoid',
          pos='0 0 0.027', size='0.043 0.054 0.06', **common)
    head = base.find(".//body[@name='head_assembly']")
    child(head, 'geom', name='head_contact', type='ellipsoid',
          pos='0.015 0 -0.005', size='0.075 0.10 0.085', **common)
    contacts = root.find('contact')
    if contacts is None:
        contacts = child(root, 'contact')
    child(contacts, 'exclude', body1='base', body2='head_assembly')
    for side, sign in (('left', 1), ('right', -1)):
        for body_name, suffix, z1, z2, radius in (
            (side+'_upper_arm', 'arm', -.01, -.052, .016),
            (side+'_forearm', 'hand', 0, -.025, .016),
            (side+'_thigh', 'thigh', -.005, -.030, .012),
            (side+'_shin', 'shin', -.005, -.029, .012),
        ):
            body = base.find(f".//body[@name='{body_name}']")
            child(body, 'geom', name=f'{side}_{suffix}_contact', type='capsule',
                  fromto=f'0 0 {z1} 0 0 {z2}', size=str(radius), **common)
        for parent, descendant in (
            ('base', side+'_upper_arm'), ('base', side+'_thigh'),
            (side+'_upper_arm', side+'_forearm'),
            (side+'_thigh', side+'_shin'), (side+'_shin', side+'_foot'),
        ):
            child(contacts, 'exclude', body1=parent, body2=descendant)

    # Visual geoms remain noncolliding; the two boxes define the sole contact.
    for side in ('left', 'right'):
        foot = base.find(f".//body[@name='{side}_foot']")
        if foot is None:
            raise ValueError(f'{side}_foot body not found')
        if foot.find(f"geom[@name='{side}_foot_contact']") is None:
            child(foot, 'geom', name=f'{side}_foot_contact', type='box',
                  pos='0.012 0 -0.014', size='0.035 0.026 0.010',
                  **common)

    sensor = root.find('sensor')
    if sensor is None:
        sensor = ET.SubElement(root, 'sensor')
    existing = {item.get('name') for item in sensor}
    for tag, name, body in (
        ('framequat', 'root_orientation', 'base'),
        ('framelinvel', 'root_linear_velocity', 'base'),
        ('frameangvel', 'root_angular_velocity', 'base'),
    ):
        if name not in existing:
            child(sensor, tag, name=name, objtype='body', objname=body)
    for side in ('left', 'right'):
        name = f'{side}_foot_force'
        if name not in existing:
            child(sensor, 'touch', name=name, site=f'{side}_foot_contact_site')
    # Touch sensors need sites. Keep them coincident with the collision boxes.
    for side in ('left', 'right'):
        foot = base.find(f".//body[@name='{side}_foot']")
        if foot.find(f"site[@name='{side}_foot_contact_site']") is None:
            child(foot, 'site', name=f'{side}_foot_contact_site', pos='0.012 0 -0.014',
                  size='0.035 0.026 0.010', type='box')

    ET.indent(root, space='  ')
    output.write_text(ET.tostring(root, encoding='unicode') + '\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    print('Generated', make_free_model())
