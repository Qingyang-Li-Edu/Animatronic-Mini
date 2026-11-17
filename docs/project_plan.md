- ### Q版人偶运动控制仿真计划（macOS · ROS 2 Jazzy + MuJoCo）

  #### 1. 目标与边界

  基于 [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini)（物理模型取自 [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground)），在 macOS（RoboStack）上跑通 `ROS 2 + MuJoCo` 位置控制闭环，完成头颈与极简面部并行动作编排，并提取关节力矩验证 STS3215 舵机负载裕量。

  - **本期范围（固定基座）**：
    - **身体控制**：复用原模型颈部、头部、天线及双腿关节（躯干固定于世界坐标系，腿部仅做下蹲/前倾等姿态配合以验证力矩）。
    - **面部扩展**：在头部刚体新增 4 个转动关节（`eye_pan`、`eye_tilt`、`eyelid_close`、`jaw_open`）。
    - **控制与编排**：用双 `JointTrajectoryController`（身体 + 面部）实现样条插值与并行动作触发。
    - **负载评估**：统计典型动作下的关节峰值/有效值力矩与角速度，核对 STS3215 及微型面部舵机选型。
  - **暂不纳入**：双足强化学习平衡步态（后续直接挂载现成 ONNX 策略）、复杂表情混合状态机、实机串口通信。

  #### 2. 系统架构与数据流

  Plaintext

  ```
  [orchestrator.py] (读取 motions.yaml，对齐时间戳并行发送 Action)
     ├── FollowJointTrajectory ──► body_jtc (头/颈/天线/腿)
     └── FollowJointTrajectory ──► face_jtc (4 个面部关节)
                                      │
                       controller_manager (100Hz, use_sim_time)
                                      │ (topic_based_hardware_interfaces)
              /mujoco/joint_commands  ▼       ▲  /mujoco/joint_states (100Hz, 含 effort)
                       [mujoco_node.py · mjpython 启动]
                        主线程: MuJoCo 物理步进 (500Hz, dt=0.002s) + 原生 Viewer
                        子线程: rclpy 订阅控制指令、发布状态与 /clock
  ```

  *注：直接复用 `/mujoco/joint_states` 录制 bag 分析力矩（内含 `position`、`velocity`、`effort`），不再单开重复的 `/chibi/actuator_effort` 话题；可视化直接使用 MuJoCo 原生 Viewer，省去 RViz2 渲染链路。*

  #### 3. 目录结构（单包精简版）

  Plaintext

  ```
  ~/Desktop/rob/chibi_ws/
  ├── third_party/
  │   └── Open_Duck_Playground/          # 存放原版 MJCF 与 mesh 资源
  └── src/
      └── chibi_sim/                     # ament_python 单包管理
          ├── mjcf/
          │   └── chibi_fixed.xml        # 基于原版 MJCF 修改：固定基座 + 新增 4 个面部关节
          ├── urdf/
          │   └── chibi_control.urdf     # 仅声明关节名、限位与 ros2_control 硬件接口标签
          ├── config/
          │   ├── controllers.yaml       # JTC 控制器参数与关节分组
          │   └── motions.yaml           # 关键帧动作库（body + face 双轨时间序列）
          ├── launch/
          │   └── sim.launch.py          # 一键拉起 mujoco_node 和 controller_manager
          ├── chibi_sim/
          │   ├── mujoco_node.py         # 物理仿真与话题桥接
          │   └── orchestrator.py        # 动作解析与并行 Action 客户端
          └── scripts/
              └── analyze_effort.py      # 读取 rosbag 绘制力矩-转速散点图
  ```

  #### 4. 实施步骤

  | **阶段**                   | **核心任务**                                                 | **交付与验收判据**                                           |
  | -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | **Step 1：环境与冒烟测试** | 1. 在 RoboStack `ros-jazzy-desktop` 环境中通过 `pip install mujoco` 安装物理引擎，检查或源码编译 [topic_based_hardware_interfaces](https://control.ros.org/rolling/doc/topic_based_hardware_interfaces/doc/index.html)。  2. 编写最小验证脚本：用 [mjpython](https://mujoco.readthedocs.io/en/stable/python.html) 启动 `launch_passive` 窗口，同时在后台线程跑通 `rclpy` 话题收发。 | 终端无报错弹出 MuJoCo 窗口，且 `ros2 topic hz` 能测到稳定频率。 |
  | **Step 2：模型改造**       | 1. 从 [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground) 拷贝 `open_duck_mini_v2` 的 MJCF 与网格。  2. 注释掉根节点 `freejoint` 改为固定基座；在头部 `body` 下添加 4 个面部 `hinge` 关节及对应 `position` 执行器（设小惯量 `1e-5` 防止发散，关闭面部几何体碰撞 `contype="0" conaffinity="0"`）。  3. 编写匹配的 `chibi_control.urdf`（仅包含 `<joint>` 与 `<ros2_control>` 标签，无需写视觉网格）。 | `mjpython -m mujoco.viewer --mjcf=chibi_fixed.xml` 能正常拖动所有身体与新增面部关节，物理步进不发散。 |
  | **Step 3：控制闭环打通**   | 1. 实现 `mujoco_node.py`：主线程每步 `mj_step` 后同步推进 `/clock`，按关节名映射将 `/mujoco/joint_commands` 写入 `data.ctrl`，并以 100Hz 发布 `/mujoco/joint_states`（将 `data.actuator_force` 填入 `effort`）。  2. 配置 `controllers.yaml`，启动 `body_jtc` 与 `face_jtc`。 | `ros2 control list_controllers` 均显示 `active`；用 `rqt_joint_trajectory_controller` 推滑块时，MuJoCo 窗口内模型平滑跟随。 |
  | **Step 4：动作编排**       | 1. 在 `motions.yaml` 定义基础动作（`nod`、`shake`、`look_around`、`crouch`、`greet`、`blink`）。  2. 实现 `orchestrator.py`：订阅字符串指令，读取 YAML 组装 `JointTrajectory` 消息，打上统一的未来时间戳（`now + 0.05s`）并行发送给两个 JTC Action Server。 | 下发 `look_around` 指令时，转头与眼球转动同步触发；单独触发 `blink` 时不打断正在执行的身体动作。 |
  | **Step 5：舵机负载校核**   | 1. 在 MJCF 中将执行器 `forcerange` 设为舵机额定力矩上限（参考 [Open Duck Mini sim2real 文档](https://github.com/apirrone/Open_Duck_Mini/blob/v2/docs/sim2real.md)）。  2. 连续触发全部预设动作并录制 `ros2 bag`，运行 `analyze_effort.py` 输出各关节峰值力矩、RMS 力矩及最大角速度。 | 全部动作下峰值力矩 < 70% 堵转力矩，RMS 力矩在额定连续工作区内，确认舵机不发生持续饱和。 |
