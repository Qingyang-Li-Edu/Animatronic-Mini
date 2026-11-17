# Chibi Sim

## Layout

| Path | Contents |
| --- | --- |
| `src/chibi_sim/` | ROS 2 package: MJCF/URDF models, controllers, motions, launch, helper scripts |
| `training/` | PPO training, evaluation and stress-test scripts; small reference checkpoints in `checkpoints/` |
| `scripts/`, `bluefish_viewer/` | Local VRM viewer, training bridge and calibration tools |
| `validation/` | Validation reports (JSON) |
| `docs/` | Character design, model notes, project plan |
| `third_party/` | Vendored upstream sources (Open Duck Playground, topic_based_hardware_interfaces) |

> The BlueFatFish VRM model is **not** included: its author forbids redistribution. Download it yourself and place it in `third_party/bluefatfish/` (see `docs/bluefatfish_model.md`). TensorBoard logs are not versioned.

ROS 2 Jazzy + MuJoCo fixed-base animatronic mini simulation. The generated model is based on the vendored Open Duck Mini v2 MJCF and its mesh assets, with a fixed base, articulated antennae, and four added face joints.

## Setup

```bash
conda env create -f environment.yml
conda activate chibi-sim
python -m mujoco.viewer --mjcf=src/chibi_sim/mjcf/chibi_fixed.xml
```

The active macOS arm64 RoboStack channels do not contain ROS 2 Jazzy packages. Use the provided Linux container for the integrated control stack:

```bash
docker build -t chibi-sim:jazzy .
docker run --rm -it --name chibi-sim chibi-sim:jazzy
```

The container builds the Jazzy `joint_state_topic_hardware_interface` from its upstream source in `third_party/topic_based_hardware_interfaces`. In a second terminal, send a motion with `docker exec -it chibi-sim bash -c 'source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && ros2 topic pub --once /chibi/motion std_msgs/msg/String "{data: look_around}"'`. Run all six motions and validate the state stream with `docker exec -it chibi-sim bash -c 'source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && python3 /ws/src/chibi_sim/scripts/smoke_test_motions.py'`. The container runs MuJoCo headless when no Linux display is attached; the local macOS viewer command above opens the model independently.

## RViz2 in your browser (macOS)

RViz2 runs with ROS 2 inside the container; no macOS ROS installation or XQuartz is needed.
MuJoCo computes physics, joint_state_broadcaster publishes `/joint_states`, and
robot_state_publisher provides the branched robot mesh model and TF to RViz2.

```bash
cd Animatronic-Mini
docker build -t chibi-sim:jazzy .
docker run -d --name chibi-web -p 127.0.0.1:6080:6080 \
  chibi-sim:jazzy bash /ws/scripts/web_visualization.sh
```

Open `http://localhost:6080/vnc.html?autoconnect=true&resize=scale`.
This is a live RViz2 desktop served through noVNC, limited to the local machine.
The saved layout uses `base_link` and shows mesh geometry, TF axes, and a grid.

Trigger a motion in another terminal:

```bash
docker exec chibi-web bash -lc 'source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && ros2 topic pub --once /chibi/motion std_msgs/msg/String "{data: look_around}"'
```

Available names: `nod`, `shake`, `look_around`, `crouch`, `greet`, `blink`.
Loop all six automatically by adding `-e CHIBI_DEMO=1` to `docker run`, or run one smoke test with:

```bash
docker exec chibi-web bash -lc 'source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && python3 /ws/src/chibi_sim/scripts/smoke_test_motions.py'
```

Pause the automatic demo with `docker exec chibi-web touch /tmp/chibi-demo.pause`; resume with
`docker exec chibi-web rm -f /tmp/chibi-demo.pause`. Pause it before sending manual commands.

Stop or restart with `docker stop chibi-web` / `docker start chibi-web`.
Read diagnostics with `docker exec chibi-web cat /tmp/rviz.log` and
`docker exec chibi-web cat /tmp/chibi-sim.log`.

The official Open Duck Playground source is pinned by the vendored checkout at `third_party/Open_Duck_Playground`. Regenerate the fixed MJCF and control URDF after source-model edits with `python src/chibi_sim/scripts/prepare_model.py`.

## Smoke tests

```bash
python -m py_compile src/chibi_sim/chibi_sim/*.py src/chibi_sim/scripts/*.py
python src/chibi_sim/scripts/analyze_effort.py --self-test
```

## BlueFatFish VRM visual preview

The downloaded BlueFatFish VRM is the only displayed appearance. MuJoCo uses a hidden physics skeleton for joints, masses and collision proxies; the local viewer wraps the VRM around those states and retargets them onto its humanoid bones:

```bash
.venv-train/bin/python scripts/serve_bluefish_viewer.py --source training
```

页面现在还提供 `站立招手`：0.8 N 留出推力 30/30 通过，120 秒连续回放通过；3 N 成功率 86.7%，仍未过强扰动门槛。

Open <http://localhost:8766/>. The model stays in the ignored `third_party/bluefatfish/` directory and is excluded from Docker builds. Personal noncommercial use only; do not redistribute the model or a container that includes it. Use `--source ros` only when connecting to a separately running ROS container. See [bluefatfish_model.md](docs/bluefatfish_model.md) for author credit and the package terms.

## Training and physics validation

Use the isolated native macOS training environment (Homebrew Python 3.12):

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-train
.venv-train/bin/python -m pip install -r training/requirements-macos.lock.txt
.venv-train/bin/python src/chibi_sim/scripts/prepare_training_models.py
.venv-train/bin/python src/chibi_sim/scripts/validate_training_physics.py
```

Train and evaluate fixed-base residual pose tracking:

```bash
.venv-train/bin/python training/train_pose.py --motion greet --timesteps 262144 \
  --workers 2 --device cpu --out training/checkpoints/ppo-safe
.venv-train/bin/python training/evaluate.py --motion greet \
  --policy training/checkpoints/ppo-safe/greet.zip \
  --out validation/reports/greet_ppo_safe.json
.venv-train/bin/python training/stress_test.py \
  --policy training/checkpoints/ppo-safe/greet.zip
```

The isolated environment resolves the conda/PyTorch OpenMP conflict without
unsafe runtime overrides. Training uses CPU and two spawn simulation workers.
The trained greet policy reduced nominal active-joint RMSE from 0.01837 rad
(reference commands) to 0.00538 rad. It passed 100 continuous cycles and ten
held-out mass/inertia and initial-state perturbations; worst RMSE was 0.00562 rad.
These results are fixed-base trajectory compensation, not autonomous locomotion.

Free-root standing and push experiments are available separately:

```bash
.venv-train/bin/python training/train_balance.py --seed 7 \
  --out training/checkpoints/balance-seed7
.venv-train/bin/python training/evaluate_balance.py \
  --policy training/checkpoints/balance-seed7/stand.zip
```

The free model uses provisional mass, inertia, friction and motor limits.
Seven physics scenarios pass, including timestep changes, parameter changes,
drop contact and whole-body fall collisions. These are simulation checks with
uncalibrated parameters; they do not establish hardware safety or walking.
See [training/README.md](training/README.md) for methods, results and limits.
