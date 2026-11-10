"""Measure the Cartesian controller on its own: no cube, no expert."""
import sys, json
import numpy as np
from panda_env import *

env = PandaPickPlace()
env.reset(0, cube_xy=(0.2, -0.4))        # cube parked out of the way
HI, LO = 0.13, 0.03
corners = [(SPAWN_X[0], SPAWN_Y[0]), (SPAWN_X[1], SPAWN_Y[0]), (SPAWN_X[1], SPAWN_Y[1]),
           (SPAWN_X[0], SPAWN_Y[1]), tuple(TARGET_XY)]
way = []
for x, y in corners:
    way += [(x, y, HI), (x, y, LO), (x, y, HI)]
t, tcp, cmd, tilt, settle = [], [], [], [], []
for w in map(np.array, way):
    for _ in range(int(3 / POLICY_DT)):
        if np.linalg.norm(w - env.x_cmd) < 1e-5:
            break
        env.step(np.r_[np.clip(w - env.x_cmd, -MAX_DELTA, MAX_DELTA), 1])
        t.append(env.d.time); tcp.append(env.tcp_pos()); cmd.append(env.x_cmd.copy())
        tilt.append(np.degrees(np.arccos(np.clip(-env.d.site_xmat[env.tcp][8], -1, 1))))
    n_move = len(t)
    for _ in range(10):      # 0.4 s settle
        env.step([0, 0, 0, 1])
        t.append(env.d.time); tcp.append(env.tcp_pos()); cmd.append(env.x_cmd.copy())
        tilt.append(np.degrees(np.arccos(np.clip(-env.d.site_xmat[env.tcp][8], -1, 1))))
    settle.append(np.linalg.norm(env.tcp_pos() - w))
tcp, cmd, t = np.array(tcp), np.array(cmd), np.array(t)
e = np.linalg.norm(tcp - cmd, axis=1)
res = dict(moves=len(way), peak_cmd_speed_m_s=MAX_DELTA / POLICY_DT,
           rms_track_mm=float(np.sqrt((e ** 2).mean()) * 1e3), peak_track_mm=float(e.max() * 1e3),
           settled_worst_mm=float(max(settle) * 1e3), settled_median_mm=float(np.median(settle) * 1e3),
           peak_tilt_deg=float(max(tilt)))
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "controller.json", "w"), indent=1)
np.savez(sys.argv[2] if len(sys.argv) > 2 else "controller_trace.npz", t=t, tcp=tcp, cmd=cmd)
