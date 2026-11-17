#!/usr/bin/env python3
"""Serve the BlueFatFish preview and its original VRM on loopback only."""
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import math
import subprocess
import threading
import time
from urllib.parse import urlsplit
from bluefish_training_bridge import TrainingBridge

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'bluefish_viewer' / 'index.html'
MODEL = ROOT / 'third_party' / 'bluefatfish' / 'BlueFatFish.vrm'
MOTIONS = {'balance', 'balance_strong', 'greet_balance', 'greet_balance_strong', 'idle', 'nod', 'shake', 'greet', 'blink', 'happy', 'look_around', 'crouch'}

# Executed in the existing ROS container. The VRM stays on the Mac.
ROS_STATE_CODE = '''
import json, sys, threading, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
stopped = threading.Event()
def watch_input():
    for _ in sys.stdin:
        pass
    stopped.set()
threading.Thread(target=watch_input, daemon=True).start()
rclpy.init()
node = Node('bluefish_visual_state_bridge')
last = 0.0
def state(msg):
    global last
    now = time.monotonic()
    if now - last < .033:
        return
    last = now
    print(json.dumps(dict(zip(msg.name, msg.position)), allow_nan=False), flush=True)
subscription = node.create_subscription(JointState, '/mujoco/joint_states', state, 10)
try:
    while rclpy.ok() and not stopped.is_set():
        rclpy.spin_once(node, timeout_sec=.2)
finally:
    node.destroy_node()
    rclpy.shutdown()
'''


class JointBridge:
    def __init__(self):
        self.lock = threading.Lock()
        self.positions = {}
        self.received_at = 0.0
        self.process = None
        self.stopped = threading.Event()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        import shlex
        command = ('source /opt/ros/jazzy/setup.bash && '
                   'source /ws/install/setup.bash && exec python3 -u -c ' +
                   shlex.quote(ROS_STATE_CODE))
        while not self.stopped.is_set():
            try:
                process = subprocess.Popen(
                    ['docker', 'exec', '-i', 'chibi-character', 'bash', '-lc', command],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, text=True, bufsize=1)
                self.process = process
                for line in process.stdout:
                    if self.stopped.is_set():
                        break
                    try:
                        q = json.loads(line)
                        if (not isinstance(q, dict) or len(q) != 24 or
                                not all(isinstance(k, str) and isinstance(v, (float, int))
                                        and math.isfinite(v) for k, v in q.items())):
                            continue
                        with self.lock:
                            self.positions = q
                            self.received_at = time.monotonic()
                    except (ValueError, TypeError):
                        continue
                process.wait(timeout=3)
            except (OSError, subprocess.SubprocessError):
                pass
            self.stopped.wait(3)

    def snapshot(self):
        with self.lock:
            age = time.monotonic() - self.received_at if self.received_at else None
            return {'connected': age is not None and age < 2,
                    'age_seconds': age, 'positions': self.positions,
                    'source': '/mujoco/joint_states', 'joint_count': len(self.positions)}

    def stop(self):
        self.stopped.set()
        process = self.process
        if process and process.poll() is None:
            process.stdin.close()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()


bridge = None


class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self._serve(send_body=False)

    def do_GET(self):
        self._serve(send_body=True)

    def do_POST(self):
        origin = self.headers.get('Origin')
        if origin and origin not in {'http://localhost:8766', 'http://127.0.0.1:8766'}:
            self.send_error(403, 'Local viewer origin required')
            return
        if self.path.startswith('/motion/'):
            name = self.path.rsplit('/', 1)[-1]
            if name not in MOTIONS:
                self.send_error(400, 'Unknown motion')
                return
            if name in {'idle', 'happy'}:
                result = {'status': 'local_preview_only', 'motion': name}
            elif isinstance(bridge, TrainingBridge):
                bridge.set_motion(name)
                result = {'status': 'queued_in_mujoco', 'motion': name}
            elif name.startswith('balance'):
                self.send_error(400, 'Balance requires --source training')
                return
            else:
                try:
                    subprocess.run(
                        ['docker', 'exec', 'chibi-character', 'bash', '-lc',
                         "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && "
                         f"ros2 topic pub --once /chibi/motion std_msgs/msg/String '{{data: {name}}}'"],
                        check=True, capture_output=True, text=True, timeout=8)
                    result = {'status': 'sent_to_mujoco', 'motion': name}
                except (OSError, subprocess.SubprocessError) as error:
                    result = {'status': 'preview_only', 'motion': name, 'error': str(error)}
            payload = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def _serve(self, send_body):
        if self.path == '/':
            path, mime = PAGE, 'text/html; charset=utf-8'
        elif self.path == '/BlueFatFish.vrm':
            path, mime = MODEL, 'model/gltf-binary'
        elif self.path == '/calibration.json':
            path, mime = ROOT/'bluefish_viewer/calibration.json', 'application/json'
        elif self.path == '/api/state':
            payload = json.dumps(bridge.snapshot(), allow_nan=False).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            if send_body:
                self.wfile.write(payload)
            return
        elif self.path == '/health':
            if not MODEL.is_file():
                self.send_error(503, 'Local VRM file is missing')
                return
            payload = (f'{{"status":"ready","bytes":{MODEL.stat().st_size},'
                       f'"sha256":"{sha256(MODEL.read_bytes()).hexdigest()}"}}').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            if send_body:
                self.wfile.write(payload)
            return
        else:
            self.send_error(404)
            return

        if not path.is_file():
            self.send_error(503, 'Local preview file is missing')
            return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(path.stat().st_size))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if send_body:
            with path.open('rb') as source:
                while block := source.read(128 * 1024):
                    self.wfile.write(block)

    def log_message(self, fmt, *args):
        if self.path == '/api/state':
            return
        print('%s - %s' % (self.address_string(), fmt % args), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['training', 'ros'], default='training')
    args = parser.parse_args()
    if not PAGE.is_file() or not MODEL.is_file():
        raise SystemExit('The viewer page or local VRM model is missing')
    bridge = TrainingBridge() if args.source == 'training' else JointBridge()
    server = ThreadingHTTPServer(('127.0.0.1', 8766), Handler)
    print(f'Open http://localhost:8766/ (downloaded VRM shell; {args.source})', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        bridge.stop()
