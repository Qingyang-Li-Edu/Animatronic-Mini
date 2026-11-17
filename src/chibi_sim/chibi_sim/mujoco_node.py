"""MuJoCo <-> ROS bridge. Runs headless or with native viewer when available."""
import math, os, threading, time
from pathlib import Path
try:
 import mujoco
except ImportError: mujoco = None
try:
 import rclpy
 from rclpy.node import Node
 from rclpy.executors import SingleThreadedExecutor
 from sensor_msgs.msg import JointState
 from rosgraph_msgs.msg import Clock
 from builtin_interfaces.msg import Time
except ImportError: rclpy = None

BODY_JOINTS = ['neck_pitch', 'head_pitch', 'head_yaw', 'head_roll', 'antenna_l', 'antenna_r',
               'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
               'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle']
FACE_JOINTS = ['eye_pan', 'eye_tilt', 'eyelid_close', 'jaw_open']
JOINTS = BODY_JOINTS + FACE_JOINTS
class Bridge(Node):
 def __init__(self, model, data):
  super().__init__('mujoco_node'); self.model=model; self.data=data; self.lock=threading.Lock()
  self.joint_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt)]
  self.commands={j:0.0 for j in self.joint_names}
  self.create_subscription(JointState, '/mujoco/joint_commands', self.command_cb, 10)
  self.pub=self.create_publisher(JointState, '/mujoco/joint_states', 10); self.clock_pub=self.create_publisher(Clock,'/clock',10)
 def command_cb(self,msg):
  with self.lock:
   for n,p in zip(msg.name,msg.position):
    if n in self.commands and math.isfinite(float(p)): self.commands[n]=float(p)
 def apply(self):
  with self.lock:
   for j,v in self.commands.items():
    jid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,j)
    aid=next((a for a in range(self.model.nu) if int(self.model.actuator_trnid[a,0])==jid),-1) if jid>=0 else -1
    if aid>=0: self.data.ctrl[aid]=v
 def publish(self):
  msg=JointState(); msg.header.stamp=sim_time(self.data.time); msg.name=self.joint_names
  msg.position=[float(self.data.qpos[self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,j)]]) for j in self.joint_names]
  msg.velocity=[float(self.data.qvel[self.model.jnt_dofadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,j)]]) for j in self.joint_names]
  msg.effort=[float(self.data.qfrc_actuator[self.model.jnt_dofadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,j)]]) for j in self.joint_names]; self.pub.publish(msg)
def sim_time(seconds):
 t=Time(); t.sec=int(seconds); t.nanosec=int((seconds-t.sec)*1_000_000_000); return t
def main():
 if mujoco is None: raise RuntimeError('Install mujoco with pip install mujoco')
 default_path=Path(__file__).parents[1]/'mjcf/chibi_fixed.xml'
 try:
  from ament_index_python.packages import get_package_share_directory
  default_path=Path(get_package_share_directory('chibi_sim'))/'mjcf/chibi_fixed.xml'
 except ImportError: pass
 path=Path(os.environ.get('CHIBI_MJCF', default_path)); model=mujoco.MjModel.from_xml_path(str(path)); data=mujoco.MjData(model)
 ros= rclpy is not None
 if ros:
  rclpy.init(); node=Bridge(model,data); executor=SingleThreadedExecutor(); executor.add_node(node)
  ros_thread=threading.Thread(target=executor.spin, name='ros-executor', daemon=True); ros_thread.start()
 viewer=None
 try:
  try:
   viewer=mujoco.viewer.launch_passive(model,data)
  except Exception: pass
  deadline=time.perf_counter()
  steps=0
  while not ros or rclpy.ok():
   node.apply() if ros else None; mujoco.mj_step(model,data)
   if ros:
    c=Clock(); c.clock=sim_time(data.time); node.clock_pub.publish(c)
    steps+=1
    if steps % max(1,round(0.01/model.opt.timestep)) == 0: node.publish()
   if viewer: viewer.sync()
   deadline+=model.opt.timestep
   time.sleep(max(0.0, deadline-time.perf_counter()))
 finally:
  if ros:
   executor.shutdown(); node.destroy_node(); rclpy.shutdown(); ros_thread.join(timeout=1.0)
if __name__=='__main__': main()
