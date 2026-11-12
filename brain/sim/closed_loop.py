"""Week 6: the fly brain drives the arm (reaching only).

One control step (40 ms), in this order:

    arm state --normalise--> s (4 signals) --Encoder--> Poisson rates on 335 JO-C/E neurons
              --BrainIO.step--> brain advanced brain_ms (default 40 ms = 1 control step)
              --Decoder--> DN-candidate rates over the last 40 ms
              --RidgeReadout--> dx, dy, dz            (gripper held open: reach only)
              --PandaPickPlace.step--> 40 ms of physics through the IK controller

The connectome is frozen. The only thing trained is the linear readout (ridge regression)
from output-neuron rates to the expert's action, so any nonlinearity the policy has must
come from the brain.

Training uses DAgger: run the current policy, label every state it visits with what the
expert would do there, refit on everything collected so far. Round 0 lets the expert
drive (half the episodes with 5 mm action noise, as in week 4); later rounds let the
brain drive. The expert is memoryless (week 4), so it can label any state.

Reach task (fixed before any brain run):
  * reached: gripper xy within 15 mm of the cube centre AND gripper below 4.2 cm
    (the expert's own grasp-ready condition: CLOSE_XY, GRASP_Z + 12 mm), held 5 steps (0.2 s)
  * travel:  gripper xy within 15 mm of the cube at any height (the travel half)
  * knocked: cube pushed more than 10 mm from its spawn point -> failure, episode stops
  * timeout: 75 steps (3 s). The expert needs 15 (median), 17 at most.

Control: BrainPolicy.feedback = False keeps the encoder on the episode's first state, so the
brain never sees the arm move. If that still reaches, the readout learned a fixed drift, not
a closed loop.
"""
import time
import numpy as np

from .interface import normalise, Decoder
from .readout import _fit_path, _standardise, _folds, ALPHAS

REACH_XY = 0.015        # m, = expert CLOSE_XY
REACH_Z = 0.042         # m, = expert GRASP_Z + 0.012
HOLD = 5                # steps the reached condition must hold
KNOCK = 0.010           # m, cube displacement that counts as knocked
REACH_MAX_STEPS = 75
OPEN = 1.0


# ---------------------------------------------------------------- the target
def reach_label(obs, expert):
    """What the expert would do here, restricted to reaching: dx, dy, dz.
    Where the expert would close the gripper, the reach is done, so the label is 'hold still'."""
    a, phase = expert(obs)
    if phase not in ('reach', 'descend'):
        return np.zeros(3), phase
    return np.asarray(a[:3], float), phase


def reached(obs):
    rel = obs[16:19]
    return bool(np.hypot(rel[0], rel[1]) < REACH_XY and obs[2] < REACH_Z)


# ---------------------------------------------------------------- the readout
class RidgeReadout:
    """Linear map, features -> action, with the ridge penalty chosen by cross-validation
    grouped by episode (same rules as week 5's readout.py)."""

    def __init__(self, alphas=ALPHAS, n_folds=5, seed=0):
        self.alphas, self.n_folds, self.seed = alphas, n_folds, seed

    def fit(self, X, Y, groups):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        mu, sd = X.mean(0), X.std(0)
        self.ok = sd > 1e-9
        self.mu, self.sd = mu[self.ok], sd[self.ok]
        err = np.zeros(len(self.alphas))
        if len(np.unique(groups)) >= self.n_folds:
            for f in _folds(groups, self.n_folds, self.seed):
                a, b = _standardise(X[~f], X[f])
                m = Y[~f].mean(0)
                for i, W in enumerate(_fit_path(a, Y[~f] - m, self.alphas)):
                    err[i] += ((b @ W + m - Y[f]) ** 2).sum()
            self.alpha = float(self.alphas[int(np.argmin(err))])
            self.cv_mse = err / len(Y)
        else:
            self.alpha = float(self.alphas[len(self.alphas) // 2])
        Z = (X[:, self.ok] - self.mu) / self.sd
        self.ym = Y.mean(0)
        self.W = _fit_path(Z, Y - self.ym, [self.alpha])[0]
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return ((X[:, self.ok] - self.mu) / self.sd) @ self.W + self.ym

    def state(self):
        return dict(ok=self.ok, mu=self.mu, sd=self.sd, W=self.W, ym=self.ym, alpha=self.alpha)


# ---------------------------------------------------------------- policies
class BrainPolicy:
    """The fly brain. act(obs) advances the brain brain_ms and returns dx, dy, dz.

    features(): rates (Hz) over the last window_ms of EVERY recorded neuron (BrainIO.record),
    so the data can be refit later with a different readout pool without rerunning the brain.
    cols: which of those columns the readout uses (default: all). raster_cols: which go in
    the video's raster."""
    name = 'brain'

    def __init__(self, io, w_scale, obs_names, cols=None, raster_cols=None, brain_ms=40,
                 window_ms=40, warmup_ms=200, readout=None):
        self.io, self.w, self.obs_names = io, w_scale, obs_names
        n = len(io.record)
        self.dec = Decoder(np.arange(n), window_ms, io.bin_ms)
        self.cols = np.arange(n) if cols is None else np.asarray(cols)
        self.raster_cols = self.cols if raster_cols is None else np.asarray(raster_cols)
        assert brain_ms >= window_ms, "the decoder window must fit in one brain advance"
        self.brain_ms, self.warmup_ms = brain_ms, warmup_ms
        self.readout = readout
        self.counts = None
        self.feedback = True     # False: the brain only ever sees the first state (control)

    def reset(self, seed, obs):
        self.io.reset(seed, self.w)
        self.s0 = normalise(obs, self.obs_names)[0]
        self.io.step(self.s0, self.warmup_ms)                              # hold at the first state

    def features(self, obs):
        s = normalise(obs, self.obs_names)[0] if self.feedback else self.s0
        self.counts, _ = self.io.step(s, self.brain_ms)
        self.s = s
        return self.dec.rates(self.counts)

    def act(self, obs):
        x = self.features(obs)
        a = np.zeros(3) if self.readout is None else self.readout.predict(x[self.cols])[0]
        return a, x


class SignalPolicy:
    """No brain: the same linear readout applied straight to the 4 encoded signals
    (noise-free). The control for 'did the brain add anything a linear map can't do'."""
    name = 'signals'

    def __init__(self, obs_names, readout=None):
        self.obs_names, self.readout = obs_names, readout
        self.counts, self.cols = None, np.arange(4)

    def reset(self, seed, obs):
        pass

    def features(self, obs):
        self.s = normalise(obs, self.obs_names)[0]
        return self.s

    def act(self, obs):
        x = self.features(obs)
        a = np.zeros(3) if self.readout is None else self.readout.predict(x)[0]
        return a, x


class ExpertPolicy:
    name = 'expert'

    def __init__(self, expert, obs_names):
        self.expert, self.obs_names, self.counts = expert, obs_names, None
        self.cols = np.arange(4)

    def reset(self, seed, obs):
        pass

    def act(self, obs):
        self.s = normalise(obs, self.obs_names)[0]
        return reach_label(obs, self.expert)[0], self.s


# ---------------------------------------------------------------- one episode
def run_episode(env, policy, expert, seed, expert_mix=0.0, action_noise=0.0,
                max_steps=REACH_MAX_STEPS, record=False, stop_on_reach=True):
    """Closed-loop reach. Every step logs the policy's features and the expert's label for
    the state actually visited (the DAgger data).
    expert_mix: probability that a step executes the expert's action instead of the policy's
    (the brain still runs and its features are still logged).
    record: also keep what a video needs (joint positions, spikes, actions)."""
    rng = np.random.default_rng(seed + 7_000_000)
    obs = env.reset(seed)
    cube0 = env.cube_pos()
    policy.reset(seed, obs)
    X, Y, A, O, PH, REC = [], [], [], [], [], []
    hold, first, travel, knocked, min_xy = 0, None, None, False, np.inf
    for t in range(max_steps):
        a_pol, x = policy.act(obs)
        label, phase = reach_label(obs, expert)
        a = label if rng.random() < expert_mix else np.asarray(a_pol, float)
        ex = a.copy()
        if action_noise:
            ex += rng.normal(0, action_noise, 3)
        X.append(x); Y.append(label); A.append(a); O.append(obs); PH.append(phase)
        if record:
            spk = policy.counts[:, policy.raster_cols].copy() \
                if getattr(policy, 'counts', None) is not None else None
            REC.append(dict(qpos=env.d.qpos.copy(), counts=spk, s=policy.s.copy(),
                            act=a.copy(), label=label.copy()))
        obs = env.step(np.r_[ex, OPEN])
        xy = float(np.hypot(*obs[16:18]))
        min_xy = min(min_xy, xy)
        if travel is None and xy < REACH_XY:
            travel = t + 1
        ok = reached(obs)
        hold = hold + 1 if ok else 0
        if hold == 1:
            first = t + 1
        if np.linalg.norm(env.cube_pos()[:2] - cube0[:2]) > KNOCK:
            knocked = True
            break
        if hold >= HOLD and stop_on_reach:
            break
    if record:
        REC.append(dict(qpos=env.d.qpos.copy(), counts=None, s=None, act=None, label=None))
    success = hold >= HOLD and not knocked
    return dict(seed=seed, success=success, steps=first if success else None,
                travel=travel is not None, travel_steps=travel, knocked=knocked,
                min_xy_mm=1000 * min_xy, final_xy_mm=1000 * float(np.hypot(*obs[16:18])),
                final_z_mm=1000 * float(obs[2]), n=len(X),
                X=np.array(X), Y=np.array(Y), A=np.array(A), obs=np.array(O), phase=np.array(PH),
                rec=REC if record else None)


# ---------------------------------------------------------------- DAgger and evaluation
def dagger(env, policy, expert, train_seeds, betas=(1.0, 0.0, 0.0, 0.0), noise=0.005,
           verbose=True, readout_kw=None, on_round=None):
    """train_seeds: one list of seeds per round (len(betas) rounds). In every round, half the
    episodes add `noise` (m per step) to the executed action. Returns (readout, data, log).
    on_round(r, data_so_far, log): called after each refit, e.g. to save a checkpoint."""
    data = dict(X=[], Y=[], g=[], round=[])
    log = []
    t0 = time.perf_counter()
    for r, (beta, seeds) in enumerate(zip(betas, train_seeds)):
        outs = []
        for i, sd in enumerate(seeds):
            o = run_episode(env, policy, expert, sd, expert_mix=beta,
                            action_noise=noise if i % 2 else 0.0)
            outs.append(o)
            data['X'].append(o['X']); data['Y'].append(o['Y'])
            data['g'].append(np.full(o['n'], sd)); data['round'].append(np.full(o['n'], r))
        X, Y, g = (np.concatenate(data[k]) for k in ('X', 'Y', 'g'))
        policy.readout = RidgeReadout(**(readout_kw or {})).fit(X[:, policy.cols], Y, g)
        row = dict(round=r, beta=beta, episodes=len(seeds), rows=len(X), alpha=policy.readout.alpha,
                   success=np.mean([o['success'] for o in outs]),
                   travel=np.mean([o['travel'] for o in outs]),
                   knocked=np.mean([o['knocked'] for o in outs]),
                   wall_s=time.perf_counter() - t0)
        log.append(row)
        if verbose:
            who = 'expert drives' if beta == 1 else (f'{policy.name} drives' if beta == 0 else f'mix {beta}')
            print(f"  round {r} ({who}): {len(seeds)} episodes, success while collecting "
                  f"{row['success']:.0%}, knocked {row['knocked']:.0%}; refit on {len(X):,} rows, "
                  f"alpha {row['alpha']:.3g}  [{row['wall_s']:.0f} s]")
        if on_round is not None:
            on_round(r, data, log)
    for k in ('X', 'Y', 'g', 'round'):
        data[k] = np.concatenate(data[k])
    return policy.readout, data, log


def evaluate(env, policy, expert, seeds, verbose=True, label=''):
    t0 = time.perf_counter()
    outs = [run_episode(env, policy, expert, sd) for sd in seeds]
    s = summarise(outs)
    if verbose:
        print(f"  {label or policy.name}: reach {s['success']}/{s['n']} ({s['success_rate']:.0%}, "
              f"95% CI {s['ci'][0]:.0%}-{s['ci'][1]:.0%}), travel {s['travel_rate']:.0%}, "
              f"knocked {s['knocked_rate']:.0%}, median time {s['median_steps']} steps  "
              f"[{time.perf_counter() - t0:.0f} s]")
    return outs, s


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarise(outs):
    n = len(outs)
    k = sum(o['success'] for o in outs)
    steps = [o['steps'] for o in outs if o['success']]
    return dict(n=n, success=int(k), success_rate=k / n, ci=wilson(k, n),
                travel_rate=float(np.mean([o['travel'] for o in outs])),
                knocked_rate=float(np.mean([o['knocked'] for o in outs])),
                median_steps=float(np.median(steps)) if steps else None,
                median_min_xy_mm=float(np.median([o['min_xy_mm'] for o in outs])),
                median_final_z_mm=float(np.median([o['final_z_mm'] for o in outs])))
