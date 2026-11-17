"""Replay real free-root policy dynamics in MuJoCo, or record an MP4 locally."""
import argparse
from pathlib import Path
import shutil
import subprocess
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from stable_baselines3 import PPO

from balance_env import CharacterBalanceEnv


def camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, .18]
    cam.distance, cam.azimuth, cam.elevation = .8, 205, -15
    return cam


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', type=Path, help='Omit to replay neutral servo commands')
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--force', type=float, default=.8)
    p.add_argument('--angle', type=float, help='Push direction in degrees')
    p.add_argument('--record', type=Path, help='MP4 file; requires ffmpeg')
    p.add_argument('--physics-debug', action='store_true', help='Show only the hidden collision skeleton')
    a = p.parse_args()
    if not a.physics_debug:
        if a.record:
            raise SystemExit('Default appearance is the downloaded VRM at http://127.0.0.1:8766/; native collision recording requires --physics-debug')
        import webbrowser
        webbrowser.open('http://127.0.0.1:8766/')
        print('Downloaded BlueFatFish shell: http://127.0.0.1:8766/; start scripts/serve_bluefish_viewer.py if needed')
        return
    torch.set_num_threads(1)
    policy = PPO.load(a.policy, device='cpu') if a.policy else None
    env = CharacterBalanceEnv()
    env.model.geom_rgba[1:, :] = [.25, .6, .8, .6]
    options = {'push_force': a.force}
    if a.angle is not None:
        options['push_angle'] = np.deg2rad(a.angle)
    obs, _ = env.reset(seed=a.seed, options=options)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    cam = camera()

    def step():
        nonlocal obs
        action = policy.predict(obs, deterministic=True)[0] if policy else np.zeros(env.action_space.shape)
        obs, _, term, trunc, info = env.step(action)
        return term or trunc, info

    if a.record:
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            raise SystemExit('ffmpeg is required for --record')
        a.record.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        encoder = subprocess.Popen([ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', '640x480', '-r', '25', '-i', '-', '-an',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(a.record)],
            stdin=subprocess.PIPE)
        try:
            for i in range(env.episode_steps):
                done, info = step()
                if (i+1) % 4 == 0 or done:
                    renderer.update_scene(env.data, camera=cam, scene_option=opt)
                    force = env.data.xfrc_applied[env.base, :3]
                    magnitude = float(np.linalg.norm(force))
                    if magnitude > 0 and renderer.scene.ngeom < renderer.scene.maxgeom:
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                            np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([1., .3, .1, 1.]))
                        end = env.data.xpos[env.base].copy()
                        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .005,
                                            end-.1*force/magnitude, end)
                        renderer.scene.ngeom += 1
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    controller = 'PPO' if policy else 'Neutral servo'
                    draw.rectangle((0, 0, 640, 40), fill=(12, 20, 32))
                    draw.text((12, 6), f'{controller} | t={env.data.time:.2f}s | tilt={info["tilt_deg"]:.2f} deg', fill='white')
                    draw.text((12, 23), f'Applied push: {magnitude:.2f} N | seed={a.seed} | provisional simulation', fill=(255, 180, 100))
                    encoder.stdin.write(np.asarray(frame).tobytes())
                if done:
                    break
        finally:
            renderer.close()
            encoder.stdin.close()
            result = encoder.wait()
            env.close()
        if result:
            raise SystemExit(f'ffmpeg failed with status {result}')
        print('Recorded', a.record, info)
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.lookat[:] = cam.lookat
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = cam.distance, cam.azimuth, cam.elevation
            viewer.opt.sitegroup[:] = 0
            for _ in range(env.episode_steps):
                if not viewer.is_running():
                    break
                start = time.monotonic()
                done, info = step()
                viewer.sync()
                time.sleep(max(0, env.dt-(time.monotonic()-start)))
                if done:
                    print(info)
                    break
        env.close()


if __name__ == '__main__':
    main()
