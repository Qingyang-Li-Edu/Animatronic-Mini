#!/usr/bin/env bash
# Run inside a running visualization container to loop through the six motions.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
until ros2 control list_controllers 2>/dev/null | grep -q 'face_jtc.*active'; do sleep 1; done
motions=${CHIBI_MOTIONS:-/ws/install/share/chibi_sim/config/motions.yaml}
while true; do
  for motion in nod shake look_around crouch greet blink; do
    while [[ -e /tmp/chibi-demo.pause ]]; do sleep 1; done
    duration=$(python3 -c 'import sys,yaml; m=yaml.safe_load(open(sys.argv[1]))["motions"][sys.argv[2]]; print(max((p[0] for t in m.values() for p in t.get("points",[])),default=0)+0.6)' "$motions" "$motion")
    ros2 topic pub --once /chibi/motion std_msgs/msg/String "{data: $motion}"
    sleep "$duration"
  done
done
