"""Checks on the week 4 deliverables. Each one could fail if something were wrong."""
import sys, json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from panda_env import *
from expert import expert, PHASES
from rollout import rollout

data = sys.argv[1]; out = sys.argv[2]
env = PandaPickPlace(); R = {}

# 1. success detector, on hand-made situations with known answers
def judged(setup):
    env.reset(0, cube_xy=(0.5, 0.0)); setup(); mujoco.mj_forward(env.m, env.d)
    return env.info()["success"]
def put(x, y, z=CUBE_HALF, vx=0):
    a = env.c_adr; env.d.qpos[a:a + 3] = x, y, z; env.d.qvel[env.cv_adr] = vx
T = TARGET_XY
det = {"cube at rest in zone, arm away (expect True)": judged(lambda: put(*T)),
       "cube at start, arm away (False)": judged(lambda: None),
       "cube 6 cm from zone centre (False)": judged(lambda: put(T[0] + 0.06, T[1])),
       "cube in zone but sliding (False)": judged(lambda: put(*T, vx=0.2)),
       "cube floating 3 cm up in zone (False)": judged(lambda: put(*T, z=CUBE_HALF + 0.03))}
R["detector"] = det
R["do_nothing_policy"] = float(np.mean([rollout(env, policy=lambda o: np.r_[0, 0, 0, 1], seed=500000 + i, max_steps=100)["success"] for i in range(20)]))
rng = np.random.default_rng(0)
R["random_policy"] = float(np.mean([rollout(env, policy=lambda o: rng.uniform(-1, 1, 4) * [MAX_DELTA] * 3 + [0] if False else np.r_[rng.uniform(-MAX_DELTA, MAX_DELTA, 3), rng.uniform(-1, 1)], seed=510000 + i)["success"] for i in range(20)]))
print(R, flush=True)

# 2. determinism
a = rollout(env, seed=7, log=True); b = rollout(env, seed=7, log=True)
R["same_seed_identical"] = bool(a["obs"].shape == b["obs"].shape and np.array_equal(a["obs"], b["obs"]))

# 3. dataset integrity
D = np.load(data); obs, act, ep = D["obs"], D["act"], D["episode"]
val = D["ep_is_val"]; ok = D["ep_success"]; seeds = D["ep_seed"]
rows_val = val[ep]
pred = np.array([expert(o.astype(float))[0] for o in obs])
mis = np.abs(pred - act).max(1) > 1e-4
R["dataset"] = dict(rows=len(obs), episodes=len(val), train_rows=int((~rows_val).sum()), val_rows=int(rows_val.sum()),
                    clean_success=float(ok[~D["ep_noisy"]].mean()), noisy_success=float(ok[D["ep_noisy"]].mean()),
                    failed_seeds=[int(s) for s in seeds[~ok]],
                    any_nan=bool(np.isnan(obs).any() or np.isnan(act).any()),
                    labels_recomputed_mismatch=float(mis.mean()),
                    seeds_overlap=int(len(set(seeds[val]) & set(seeds[~val]))),
                    delta_in_bounds=bool(np.abs(act[:, :3]).max() <= MAX_DELTA + 1e-6),
                    frac_steps_saturated=float((np.abs(act[:, :3]).max(1) >= MAX_DELTA - 1e-6).mean()),
                    phase_share={p: float((D["phase"] == i).mean()) for i, p in enumerate(PHASES)},
                    cube_x_range=[float(D["ep_cube_xy"][:, 0].min()), float(D["ep_cube_xy"][:, 0].max())],
                    cube_y_range=[float(D["ep_cube_xy"][:, 1].min()), float(D["ep_cube_xy"][:, 1].max())],
                    ep_len_median=float(np.median(D["ep_len"])))
print(R["dataset"], flush=True)

# 4. replay: logged executed actions, from the logged seed, must reproduce the logged outcome
Xe = D["act_exec"]; errs = []
for e in np.flatnonzero(val):
    env.reset(int(seeds[e]))
    for a_ in Xe[ep == e]:
        env.step(a_)
    errs.append(np.linalg.norm(env.cube_pos() - D["ep_final_cube"][e]))
R["replay_max_cube_error_mm"] = float(max(errs) * 1e3)
print("replay", R["replay_max_cube_error_mm"], flush=True)

# 5. signed placement bias, from all clean successful episodes
c = D["ep_final_cube"][ok & ~D["ep_noisy"]][:, :2] - TARGET_XY
R["place_bias_mm"] = (c.mean(0) * 1e3).tolist()

# 6. can the data actually be imitated? train on train episodes, fly closed loop on new seeds
use = ok[ep]
sc_y = np.r_[MAX_DELTA, MAX_DELTA, MAX_DELTA, 1.0]
def fit(kind, mask):
    X_s = StandardScaler().fit(obs[mask]); X = X_s.transform(obs[mask]); y = act[mask] / sc_y
    if kind == "ridge":
        mdl = Ridge(alpha=1.0).fit(X, y)
    else:
        mdl = MLPRegressor(hidden_layer_sizes=(256, 256), early_stopping=True, batch_size=512, max_iter=150, random_state=0).fit(X, y)
    pol = lambda o: mdl.predict(X_s.transform(o[None]))[0] * sc_y
    return mdl, X_s, pol
tr = ~rows_val & use; va = rows_val & use
clean_row = ~D["ep_noisy"][ep]
models = {"ridge, linear (clean+noisy)": ("ridge", tr), "MLP (clean only)": ("mlp", tr & clean_row), "MLP (clean+noisy)": ("mlp", tr)}
R["imitation"] = {}
for name, (kind, mask) in models.items():
    mdl, xs, pol = fit(kind, mask)
    r2 = 1 - ((mdl.predict(xs.transform(obs[va])) - act[va] / sc_y) ** 2).sum(0) / ((act[va] / sc_y - (act[va] / sc_y).mean(0)) ** 2).sum(0)
    rs = [rollout(env, policy=pol, seed=400000 + i) for i in range(100)]
    k = sum(x["success"] for x in rs)
    R["imitation"][name] = dict(train_rows=int(mask.sum()), val_r2=[float(v) for v in r2], closed_loop_success=k / 100,
                                median_place_mm=float(np.median([x["place_error"] for x in rs if x["success"]]) * 1e3) if k else None)
    print(name, R["imitation"][name], flush=True)
json.dump(R, open(f"{out}/checks.json", "w"), indent=1, default=float)
