"""Send synchronized body and face trajectories for named motions."""
import os
import yaml

try:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from std_msgs.msg import String
except ImportError:
    rclpy = None


class Orchestrator(Node):
    def __init__(self, motions):
        super().__init__('orchestrator')
        self.motions = motions
        self.body = ActionClient(self, FollowJointTrajectory, '/body_jtc/follow_joint_trajectory')
        self.face = ActionClient(self, FollowJointTrajectory, '/face_jtc/follow_joint_trajectory')
        self.create_subscription(String, '/chibi/motion', self.on_motion, 10)

    @staticmethod
    def trajectory(track, stamp):
        names = track.get('joint_names', [])
        rows = track.get('points', [])
        if not names or not rows:
            return None
        from trajectory_msgs.msg import JointTrajectory
        traj = JointTrajectory()
        traj.header.stamp = stamp
        traj.joint_names = names
        for row in rows:
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in row[1:]]
            sec = int(float(row[0]))
            point.time_from_start = Duration(sec=sec, nanosec=int((float(row[0])-sec)*1_000_000_000))
            traj.points.append(point)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        return goal

    def on_motion(self, message):
        motion = self.motions.get(message.data)
        if motion is None:
            self.get_logger().warning(f'Unknown motion: {message.data}')
            return
        stamp = self.get_clock().now().to_msg()
        stamp.nanosec += 50_000_000
        if stamp.nanosec >= 1_000_000_000:
            stamp.sec += 1
            stamp.nanosec -= 1_000_000_000
        for client, key in ((self.body, 'body'), (self.face, 'face')):
            goal = self.trajectory(motion.get(key, {}), stamp)
            if goal is None:
                continue
            if not client.server_is_ready():
                self.get_logger().warning(f'{key}_jtc action server is not ready')
                continue
            future = client.send_goal_async(goal)
            future.add_done_callback(lambda result, label=key: self.on_goal_response(result, label))

    def on_goal_response(self, future, label):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'{label}_jtc goal request failed: {exc}')
            return
        if not goal_handle.accepted:
            self.get_logger().warning(f'{label}_jtc rejected the trajectory')
            return
        self.get_logger().info(f'{label}_jtc accepted the trajectory')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda result, name=label: self.on_goal_result(result, name))

    def on_goal_result(self, future, label):
        try:
            response = future.result()
            self.get_logger().info(f'{label}_jtc trajectory finished with status {response.status}')
        except Exception as exc:
            self.get_logger().error(f'{label}_jtc result failed: {exc}')


def main():
    if rclpy is None:
        raise RuntimeError('ROS 2 rclpy, control_msgs, and trajectory_msgs are required')
    from ament_index_python.packages import get_package_share_directory
    from pathlib import Path
    default = Path(get_package_share_directory('chibi_sim')) / 'config' / 'motions.yaml'
    path = Path(os.environ.get('CHIBI_MOTIONS', default))
    with open(path, encoding='utf-8') as stream:
        motions = yaml.safe_load(stream)['motions']
    rclpy.init()
    node = Orchestrator(motions)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
