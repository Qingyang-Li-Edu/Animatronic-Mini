from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
 return LaunchDescription([
  DeclareLaunchArgument('use_sim_time', default_value='true'),
  Node(package='chibi_sim', executable='mujoco_node', name='mujoco_node', output='screen'),
  Node(package='robot_state_publisher', executable='robot_state_publisher',
       parameters=[{'robot_description': Command(['cat ', os.environ.get('CHIBI_URDF', PathJoinSubstitution([FindPackageShare('chibi_sim'),'urdf','chibi_control.urdf']))]),
                    'use_sim_time': LaunchConfiguration('use_sim_time')}], output='screen'),
  Node(package='controller_manager', executable='ros2_control_node', parameters=[os.environ.get('CHIBI_CONTROLLERS', PathJoinSubstitution([FindPackageShare('chibi_sim'),'config','controllers.yaml']))], output='screen'),
  Node(package='controller_manager', executable='spawner', arguments=['body_jtc','--controller-manager','/controller_manager'], output='screen'),
  Node(package='controller_manager', executable='spawner', arguments=['face_jtc','--controller-manager','/controller_manager'], output='screen'),
  Node(package='controller_manager', executable='spawner', arguments=['joint_state_broadcaster','--controller-manager','/controller_manager'], output='screen'),
  Node(package='chibi_sim', executable='orchestrator', output='screen', parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}])])
