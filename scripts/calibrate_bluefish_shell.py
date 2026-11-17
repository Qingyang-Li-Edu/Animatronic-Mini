"""Inspect downloaded VRM proportions and verify simulator-to-avatar joint axes."""
import hashlib
import json
from pathlib import Path
import struct
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VRM = ROOT/'third_party/bluefatfish/BlueFatFish.vrm'
PHYSICS = ROOT/'src/chibi_sim/mjcf/bluefish_physics_free.xml'
C = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])


def rotation(axis, angle):
    quat, mat = np.empty(4), np.empty(9)
    mujoco.mju_axisAngle2Quat(quat, np.eye(3)[axis], angle)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def calibrate():
    raw = VRM.read_bytes()
    size = struct.unpack_from('<I', raw, 12)[0]
    document = json.loads(raw[20:20+size])
    nodes = document['nodes']
    parents = {c:i for i,n in enumerate(nodes) for c in n.get('children', [])}
    def world(index):
        node, mat = nodes[index], np.eye(4)
        if 'matrix' in node:
            mat = np.asarray(node['matrix']).reshape(4, 4).T
        else:
            x, y, z, w = node.get('rotation', [0., 0., 0., 1.])
            r = np.empty(9)
            mujoco.mju_quat2Mat(r, np.array([w, x, y, z]))
            mat[:3, :3] = r.reshape(3, 3) @ np.diag(node.get('scale', [1., 1., 1.]))
            mat[:3, 3] = node.get('translation', [0., 0., 0.])
        return world(parents[index]) @ mat if index in parents else mat
    human = document['extensions']['VRMC_vrm']['humanoid']['humanBones']
    points = {n:world(v['node'])[:3, 3] for n,v in human.items()}
    model = mujoco.MjModel.from_xml_path(str(PHYSICS))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    def body(name):
        return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()
    leg_ratios = []
    for side in ('left', 'right'):
        avatar = sum(np.linalg.norm(points[side+b]-points[side+a])
                     for a,b in [('UpperLeg','LowerLeg'),('LowerLeg','Foot')])
        physical = sum(np.linalg.norm(body(side+b)-body(side+a))
                       for a,b in [('_thigh','_shin'),('_shin','_foot')])
        leg_ratios.append(float(avatar/physical))
    # Test the browser's rotation composition against actual MJCF transforms.
    rng, maximum = np.random.default_rng(1127), 0.
    rest_arm = {'left': -np.pi/2, 'right': np.pi/2}
    for _ in range(40):
        mujoco.mj_resetData(model, data)
        q = {}
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                q[name] = float(rng.uniform(max(model.jnt_range[i,0], -.25), min(model.jnt_range[i,1], .25)))
                data.qpos[model.jnt_qposadr[i]] = q[name]
        mujoco.mj_forward(model, data)
        for side in ('left', 'right'):
            thigh = rotation(1,q[side+'_hip_yaw']) @ rotation(2,q[side+'_hip_roll']) @ rotation(0,q[side+'_hip_pitch'])
            foot = thigh @ rotation(0,q[side+'_knee']) @ rotation(0,q[side+'_ankle'])
            arm = rotation(2,q[side+'_shoulder']+rest_arm[side])
            elbow = arm @ rotation(1,q[side+'_elbow']*(1 if side=='left' else -1))
            for name, shown, rest in ((side+'_thigh',thigh,np.eye(3)),
                                      (side+'_foot',foot,np.eye(3)),
                                      (side+'_forearm',elbow,rotation(2,rest_arm[side]))):
                i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                expected = C @ data.xmat[i].reshape(3,3) @ C.T @ rest
                maximum = max(maximum,float(np.max(np.abs(expected-shown))))
        head = (rotation(0,q['neck_pitch']) @ rotation(0,q['head_pitch']) @
                rotation(1,q['head_yaw']) @ rotation(2,q['head_roll']))
        i = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'head_assembly')
        maximum = max(maximum,float(np.max(np.abs(C @ data.xmat[i].reshape(3,3) @ C.T-head))))
    assert maximum < 1e-10, maximum
    mujoco.mj_resetData(model, data)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    result = {'status':'passed', 'model_sha256':hashlib.sha256(raw).hexdigest(),
              'coordinate_matrix':C.tolist(), 'arm_rest_z_rad':rest_arm,
              'elbow_axis':'Y', 'elbow_sign':{'left':1,'right':-1},
              'hip_euler_order':'YZX', 'head_euler_order':'XYZ',
              'avatar_units_per_physics_m':float(np.mean(leg_ratios)),
              'physics_reference_root':body('base').tolist(),
              'physics_reference_feet':{s:body(s+'_foot').tolist() for s in ('left','right')},
              'avatar_rest_bones':{n:p.tolist() for n,p in points.items() if n in (
                  'hips','spine','head','leftUpperLeg','leftLowerLeg','leftFoot',
                  'rightUpperLeg','rightLowerLeg','rightFoot')},
              'rotation_test_poses':40, 'max_rotation_matrix_error':maximum,
              'limits':['length-based shell retargeting; avatar and collision proxies have different proportions',
                        'no foot pinning or hidden base reset to conceal slip',
                        'no claim that the avatar mesh is the measured physical collision surface']}
    return result


if __name__ == '__main__':
    result = calibrate()
    for path in [ROOT/'bluefish_viewer/calibration.json', ROOT/'validation/reports/retarget_calibration.json']:
        path.write_text(json.dumps(result, indent=2)+'\n')
    print('PASS',result['rotation_test_poses'],'poses; max matrix error',result['max_rotation_matrix_error'])
    print('avatar/physics leg length scale',result['avatar_units_per_physics_m'])
