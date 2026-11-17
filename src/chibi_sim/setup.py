from setuptools import setup
from glob import glob
import os
package_name = 'chibi_sim'
setup(name=package_name, version='0.1.0', packages=[package_name], data_files=[
 ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
 ('share/' + package_name, ['package.xml']),
 ('share/' + package_name + '/config', glob('config/*.yaml')),
 ('share/' + package_name + '/launch', glob('launch/*.py')),
 ('share/' + package_name + '/mjcf', glob('mjcf/*.xml')),
 ('share/' + package_name + '/mjcf/assets', glob('mjcf/assets/*')),
 ('share/' + package_name + '/urdf', glob('urdf/*.urdf')),
 ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
], install_requires=['setuptools','pyyaml'], entry_points={'console_scripts': [
 'mujoco_node = chibi_sim.mujoco_node:main', 'orchestrator = chibi_sim.orchestrator:main']})
