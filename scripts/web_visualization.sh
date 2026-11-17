#!/usr/bin/env bash
# Run inside the container. Publish port 6080 to host loopback only.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /opt/chibi-venv/bin/activate
source /ws/install/setup.bash
share=/ws/install/share/chibi_sim
case "${CHIBI_VARIANT:-duck}" in
  duck) rviz_config="$share/rviz/chibi.rviz" ;;
  character)
    export CHIBI_MJCF="$share/mjcf/bluefish_physics.xml"
    export CHIBI_URDF="$share/urdf/bluefish_physics.urdf"
    export CHIBI_CONTROLLERS="$share/config/controllers_character.yaml"
    export CHIBI_MOTIONS="$share/config/motions_character.yaml"
    export CHIBI_EXPECTED_JOINTS=24
    rviz_config="$share/rviz/character.rviz"
    ;;
  *) echo 'CHIBI_VARIANT must be duck or character' >&2; exit 2 ;;
esac
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export QT_X11_NO_MITSHM=1
export XDG_RUNTIME_DIR=/tmp/chibi-runtime
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
pids=()
cleanup() { trap - EXIT INT TERM; kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM
Xvfb :99 -screen 0 1440x900x24 -nolisten tcp > /tmp/xvfb.log 2>&1 & pids+=("$!")
for i in {1..50}; do [[ -S /tmp/.X11-unix/X99 ]] && break; sleep 0.1; done
openbox > /tmp/openbox.log 2>&1 & pids+=("$!")
x11vnc -display :99 -localhost -rfbport 5900 -forever -shared -nopw > /tmp/x11vnc.log 2>&1 & pids+=("$!")
websockify --web=/usr/share/novnc 6080 localhost:5900 > /tmp/websockify.log 2>&1 & pids+=("$!")
ros2 launch chibi_sim sim.launch.py > /tmp/chibi-sim.log 2>&1 & pids+=("$!")
rviz2 -d "$rviz_config" --ros-args -p use_sim_time:=true > /tmp/rviz.log 2>&1 & pids+=("$!")
if [[ "${CHIBI_DEMO:-0}" == 1 ]]; then
  bash /ws/scripts/demo_motions.sh > /tmp/chibi-demo.log 2>&1 & pids+=("$!")
fi
echo 'RViz2: http://localhost:6080/vnc.html?autoconnect=true&resize=scale'
wait -n "${pids[@]}"
