#!/usr/bin/env python3
"""Inspect a local VRM/GLB without executing embedded content or modifying it."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def inspect_model(path):
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError('File is too short to be GLB/VRM')
    magic, version, length = struct.unpack_from('<4sII', data)
    if magic != b'glTF' or version != 2 or length != len(data):
        raise ValueError('Expected a complete GLB 2.0 container')
    offset, document, chunks = 12, None, []
    while offset < length:
        if offset + 8 > length:
            raise ValueError('Truncated chunk header')
        size, kind = struct.unpack_from('<II', data, offset)
        offset += 8
        if size % 4 or offset + size > length:
            raise ValueError('Invalid chunk size')
        if not chunks and kind != 0x4E4F534A:
            raise ValueError('GLB JSON must be the first chunk')
        if kind == 0x4E4F534A:
            if document is not None:
                raise ValueError('Multiple JSON chunks')
            document = json.loads(data[offset:offset + size].decode('utf-8'))
        chunks.append(kind)
        offset += size
    if document is None:
        raise ValueError('Missing GLB JSON chunk')
    ext = document.get('extensions', {})
    if 'VRMC_vrm' in ext:
        vrm = ext['VRMC_vrm']
        revision = '1.x'
        bones = vrm.get('humanoid', {}).get('humanBones', {})
        expressions = vrm.get('expressions', {})
        expression_names = sorted(set(expressions.get('preset', {})) |
                                  set(expressions.get('custom', {})))
    elif 'VRM' in ext:
        vrm = ext['VRM']
        revision = '0.x'
        bones = {bone['bone']: {'node': bone['node']} for bone in
                 vrm.get('humanoid', {}).get('humanBones', [])}
        expression_names = [group.get('name', group.get('presetName')) for group
                            in vrm.get('blendShapeMaster', {}).get('blendShapeGroups', [])]
    else:
        raise ValueError('GLB contains no VRM humanoid extension')
    nodes = document.get('nodes', [])
    for name, bone in bones.items():
        index = bone.get('node')
        if not isinstance(index, int) or not 0 <= index < len(nodes):
            raise ValueError('Invalid humanoid node for ' + name)
    meshes = document.get('meshes', [])
    required = ['hips', 'head', 'leftUpperArm', 'rightUpperArm',
                'leftLowerArm', 'rightLowerArm', 'leftUpperLeg', 'rightUpperLeg',
                'leftLowerLeg', 'rightLowerLeg', 'leftFoot', 'rightFoot']
    return {
        'file': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(),
        'size_bytes': len(data), 'vrm_revision': revision,
        'metadata': vrm.get('meta', {}),
        'humanoid_bones': bones,
        'missing_motion_bones': [name for name in required if name not in bones],
        'expressions': expression_names,
        'meshes': len(meshes), 'skins': len(document.get('skins', [])),
        'materials': len(document.get('materials', [])),
        'images': len(document.get('images', [])),
        'external_resources': [item['uri'] for category in ('buffers', 'images')
                               for item in document.get(category, [])
                               if 'uri' in item and not item['uri'].startswith('data:')],
        'extensions_used': document.get('extensionsUsed', []),
        'notice': 'Inspection only; no rig conversion or simulator replacement performed.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(inspect_model(args.model), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, 'Model inspection failed: ' + str(error) + '\n')
