"""Which part of the dataset should imitators train on? Repeats the MLP test over training seeds,
tested closed loop both clean and with 5 mm/step action noise (a stand-in for a noisy decoder)."""
import sys, json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from panda_env import *
from rollout import rollout

D = np.load(sys.argv[1]); out = sys.argv[2]; subset = sys.argv[3]
obs, act, ep = D["obs"], D["act"], D["episode"]
tr = ~D["ep_is_val"][ep] & D["ep_success"][ep]
if subset == "clean": tr &= ~D["ep_noisy"][ep]
sc_y = np.r_[MAX_DELTA, MAX_DELTA, MAX_DELTA, 1.0]
env = PandaPickPlace(); rows = []
for s in range(3):
    xs = StandardScaler().fit(obs[tr])
    mdl = MLPRegressor(hidden_layer_sizes=(256, 256), early_stopping=True, batch_size=512, max_iter=150,
                       random_state=s).fit(xs.transform(obs[tr]), act[tr] / sc_y)
    pol = lambda o: mdl.predict(xs.transform(o[None]))[0] * sc_y
    a = np.mean([rollout(env, policy=pol, seed=400000 + i)["success"] for i in range(100)])
    b = np.mean([rollout(env, policy=pol, seed=410000 + i, action_noise=0.005)["success"] for i in range(100)])
    rows.append(dict(seed=s, clean=float(a), noisy=float(b))); print(subset, rows[-1], flush=True)
json.dump(rows, open(f"{out}/imitation_{subset}.json", "w"), indent=1)
