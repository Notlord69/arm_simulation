import sys
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from panda_env import *
from expert import PHASES
from rollout import rollout
res = sys.argv[1]
S, T1, T2, B, O, G = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": T2, "axes.labelcolor": T2, "xtick.color": T2, "ytick.color": T2,
                     "text.color": T1, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
                     "axes.titlelocation": "left", "figure.facecolor": S, "axes.facecolor": S, "legend.frameon": False})
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True,
                       gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))

c = np.load(f"{res}/controller_trace.npz"); t = c["t"] - c["t"][0]; m = t < 4
ax[0].plot(t[m], c["cmd"][m, 2] * 100, color=B, lw=2, label="commanded")
ax[0].plot(t[m], c["tcp"][m, 2] * 100, color=O, lw=2, ls="--", label="measured")
ax[0].set(title="Controller follows the target: gripper height", xlabel="time (s)", ylabel="z (cm)")
ax[0].legend(loc="upper right"); ax[0].grid(axis="y", color=G)

env = PandaPickPlace(); r = rollout(env, seed=100003, log=True)
tt = np.arange(len(r["obs"])) * POLICY_DT
ax[1].plot(tt, r["obs"][:, 2] * 100, color=B, lw=2, label="gripper")
ax[1].plot(tt, r["obs"][:, 10] * 100, color=O, lw=2, label="cube")
p = r["phase"]; edges = np.r_[0, np.flatnonzero(np.diff(p)) + 1, len(p)]
for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
    if i % 2: ax[1].axvspan(a * POLICY_DT, b * POLICY_DT, color="#f1f0ec", lw=0, zorder=0)
    if (b - a) >= 3:
        ax[1].text((a + b) / 2 * POLICY_DT, 37.5, PHASES[p[a]], ha="center", va="top", fontsize=8, color=T2, rotation=90)
ax[1].set(title="One expert episode", xlabel="time (s)", ylabel="height above table (cm)", ylim=(0, 38))
ax[1].legend(loc="center right")

h = np.load(f"{res}/heldout.npz")
sc = ax[2].scatter(h["xy"][:, 0], h["xy"][:, 1], c=h["place"] * 1e3, cmap="Blues", vmin=0, vmax=6, s=26, edgecolors=T2, linewidths=0.2)
th = np.linspace(0, 2 * np.pi, 80)
ax[2].plot(TARGET_XY[0] + ZONE_RADIUS * np.cos(th), TARGET_XY[1] + ZONE_RADIUS * np.sin(th), color=O, lw=2)
ax[2].text(TARGET_XY[0], TARGET_XY[1] + 0.065, "target zone", color=T2, ha="center")
ax[2].set_ylim(-0.27, 0.37)
ax[2].set(title="Held-out starts: 500 of 500 placed", xlabel="x (m)", ylabel="y (m)", aspect="equal", xlim=(0.32, 0.63), anchor="W")
fig.colorbar(sc, ax=ax[2], label="placement error (mm)", shrink=0.85)
fig.savefig(f"{res}/week4_summary.png", dpi=170)
