FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV PATH=/opt/chibi-venv/bin:$PATH
WORKDIR /ws

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-colcon-common-extensions python3-venv python3-pip \
    ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
    ros-jazzy-rosbag2-py \
    libgl1 libglfw3 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -qq && apt-get install -y --no-install-recommends ros-jazzy-ros2-control-cmake \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -qq && apt-get install -y --no-install-recommends ros-jazzy-robot-state-publisher \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /opt/chibi-venv \
    && pip install --no-cache-dir 'mujoco>=3.2,<4' 'numpy>=1.24,<2' 'pyyaml>=6' 'matplotlib>=3.8'
ENV COLCON_PYTHON_EXECUTABLE=/opt/chibi-venv/bin/python3
ENV PYTHONPATH=/opt/chibi-venv/lib/python3.12/site-packages

RUN apt-get update -qq && apt-get install -y --no-install-recommends ros-jazzy-rviz2 xvfb x11vnc novnc websockify openbox fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY . /ws
RUN /bin/bash -c 'source /opt/ros/jazzy/setup.bash && source /opt/chibi-venv/bin/activate && colcon build --merge-install --symlink-install --base-paths src third_party/topic_based_hardware_interfaces/joint_state_topic_hardware_interface'

CMD ["bash", "-c", "source /opt/ros/jazzy/setup.bash && source /opt/chibi-venv/bin/activate && source /ws/install/setup.bash && ros2 launch chibi_sim sim.launch.py"]
