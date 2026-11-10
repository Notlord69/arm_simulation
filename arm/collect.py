"""Log (state, action) pairs for imitation.

Half the episodes are clean. Half add noise to the executed motion while logging the
clean expert label (the DART trick). A pure clean dataset only shows the perfect path,
so an imitator that drifts off it has no examples of how to get back.
Train and val are split by whole episode, with separate seed ranges.
"""
import sys, json
import numpy as np
from panda_env import *
from expert import PHASES
from rollout import rollout

NOISE = 0.005     # m per step, a level the expert handled on dev seeds
out = sys.argv[1] if len(sys.argv) > 1 else "pick_place_expert.npz"
env = PandaPickPlace()
plan = [("train", 200000, 800), ("val", 300000, 200)]
O, A, X, P, EP, EPS = [], [], [], [], [], []
ep = 0
for split, base, n in plan:
    for i in range(n):
        noisy = i % 2 == 1
        r = rollout(env, seed=base + i, log=True, action_noise=NOISE if noisy else 0.0)
        L = len(r["obs"])
        O.append(r["obs"]); A.append(r["act"]); X.append(r["act_exec"]); P.append(r["phase"])
        EP.append(np.full(L, ep, np.int32))
        EPS.append((split == "val", noisy, base + i, r["success"], L, *r["cube_xy"], r["place_error"], *r["final_cube"]))
        ep += 1
    print(split, "done", flush=True)
E = np.array(EPS, float)
np.savez_compressed(out, obs=np.concatenate(O), act=np.concatenate(A), act_exec=np.concatenate(X),
                    phase=np.concatenate(P), episode=np.concatenate(EP),
                    ep_is_val=E[:, 0].astype(bool), ep_noisy=E[:, 1].astype(bool), ep_seed=E[:, 2].astype(np.int64),
                    ep_success=E[:, 3].astype(bool), ep_len=E[:, 4].astype(int), ep_cube_xy=E[:, 5:7],
                    ep_place_error=E[:, 7], ep_final_cube=E[:, 8:11],
                    obs_names=np.array(OBS_NAMES), act_names=np.array(ACT_NAMES), phase_names=np.array(PHASES),
                    noise_m=NOISE, policy_dt=POLICY_DT)
print("rows", sum(map(len, O)), "success", E[:, 3].mean())
