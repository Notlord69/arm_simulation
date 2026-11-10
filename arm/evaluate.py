"""Expert success rate on seeds never used while writing the expert (100000+)."""
import sys, json
import numpy as np
from panda_env import *
from rollout import rollout

env = PandaPickPlace()
out = sys.argv[1] if len(sys.argv) > 1 else "."


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def summary(rs):
    k, n = sum(r["success"] for r in rs), len(rs)
    st = [r["steps"] for r in rs if r["success"]]
    pe = [r["place_error"] * 1e3 for r in rs if r["success"]]
    return dict(n=n, successes=int(k), rate=k / n, ci95=wilson(k, n),
                median_steps=float(np.median(st)) if st else None,
                p95_steps=float(np.percentile(st, 95)) if st else None,
                median_place_mm=float(np.median(pe)) if pe else None,
                p95_place_mm=float(np.percentile(pe, 95)) if pe else None,
                max_place_mm=float(np.max(pe)) if pe else None)


SWEEP_ONLY = len(sys.argv) > 2
res = json.load(open(f"{out}/expert_results.json")) if SWEEP_ONLY else {}
if not SWEEP_ONLY:
    rs = [rollout(env, seed=100000 + i) for i in range(500)]
    res["heldout_500"] = summary(rs)
    np.savez(f"{out}/heldout.npz", xy=np.array([r["cube_xy"] for r in rs]),
             success=np.array([r["success"] for r in rs]),
             steps=np.array([r["steps"] or -1 for r in rs]),
             place=np.array([r["place_error"] for r in rs]))
    gx, gy = np.linspace(*SPAWN_X, 11), np.linspace(*SPAWN_Y, 9)
    g = [[rollout(env, seed=110000 + i * 100 + j, cube_xy=(x, y)) for j, y in enumerate(gy)]
         for i, x in enumerate(gx)]
    res["grid_99"] = summary([r for row in g for r in row])
    np.savez(f"{out}/grid.npz", gx=gx, gy=gy,
             success=np.array([[r["success"] for r in row] for row in g]),
             steps=np.array([[r["steps"] or -1 for r in row] for row in g]),
             place=np.array([[r["place_error"] for r in row] for row in g]))

stress = {}
m0 = env.m
def sweep(name, **kw):
    stress[name] = summary([rollout(env, seed=120000 + i, **kw) for i in range(100)])
    print(name, stress[name]["rate"], flush=True)

for v in (0.002, 0.005, 0.01, 0.02): sweep(f"action noise {v*1e3:.0f} mm/step", action_noise=v)
for v in (0.002, 0.005): sweep(f"cube sensing noise {v*1e3:.0f} mm", obs_noise=v)
for y in (15, 30, 45): sweep(f"cube yaw {y} deg", cube_yaw=np.radians(y))
bid = env.cube_body
mass0, inertia0 = m0.body_mass[bid].copy(), m0.body_inertia[bid].copy()
for sc in (3.0, 5.0, 10.0):
    m0.body_mass[bid] = mass0 * sc; m0.body_inertia[bid] = inertia0 * sc
    mujoco.mj_setConst(m0, env.d)      # masses are cached, so they must be recomputed after an edit
    sweep(f"cube mass {mass0*sc*1e3:.0f} g")
m0.body_mass[bid] = mass0; m0.body_inertia[bid] = inertia0; mujoco.mj_setConst(m0, env.d)
saved = m0.geom_friction.copy()
for mu in (0.5, 0.3):
    # contact friction is the larger of the two geoms, so lower everything
    m0.geom_friction[:, 0] = np.minimum(saved[:, 0], mu)
    sweep(f"friction {mu}")
m0.geom_friction[:] = saved
res["stress"] = stress
json.dump(res, open(f"{out}/expert_results.json", "w"), indent=1)
