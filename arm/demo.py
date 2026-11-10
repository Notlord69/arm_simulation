"""Watch the expert.
  python demo.py                 live viewer (run this on the laptop)
  python demo.py out.mp4 3       render 3 episodes to a video, no display needed
"""
import sys, time
import numpy as np
import mujoco
from panda_env import *
from expert import expert

env = PandaPickPlace()
if len(sys.argv) > 1:
    import imageio
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    r = mujoco.Renderer(env.m, 480, 640)
    cam = mujoco.MjvCamera(); cam.lookat[:] = 0.3, 0.05, 0.2; cam.distance = 1.7; cam.azimuth = 150; cam.elevation = -22
    frames = []
    for s in range(n):
        o = env.reset(600000 + s)
        for t in range(75):
            o = env.step(expert(o)[0])
            r.update_scene(env.d, cam); frames.append(r.render().copy())
    imageio.mimsave(sys.argv[1], frames, fps=25, macro_block_size=None)
else:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(env.m, env.d) as v:
        s = 0
        while v.is_running():
            o = env.reset(600000 + s); s += 1
            for t in range(75):
                t0 = time.time(); o = env.step(expert(o)[0]); v.sync()
                time.sleep(max(0, POLICY_DT - (time.time() - t0)))
