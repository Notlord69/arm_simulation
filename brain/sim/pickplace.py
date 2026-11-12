"""Week 7: full pick-and-place with the fly brain, and the baselines it is compared with.

One control step (40 ms), the same loop as week 6 with three changes:

    arm state --normalise8--> s (8 signals) --Encoder--> Poisson rates on the 335 JO-C/E neurons
              --BrainIO.step--> brain advanced brain_ms
              --Decoder--> rates of the readout neurons over the last window_ms
              --ActionReadout--> dx, dy, dz, grip            (grip > 0 open, <= 0 close)
              --PandaPickPlace.step--> 40 ms of physics through the IK controller

  1. 8 signals instead of 4 (the week 5 decision): adds goal - cube x, y, gripper height and
     cube height, the information the grasp, carry and place phases need.
  2. A 4th output, grip. The readout predicts the expert's +-1 grip label; its sign opens or
     closes the gripper. Motion (dx, dy, dz) and grip get separate ridge fits, so each gets its
     own penalty: grip's variance (about 1) would otherwise swamp the motion axes (about 1e-4).
  3. The success test is the week 4 physics test (cube in the zone, at rest, released, gripper
     6 cm clear, held 10 steps), not the week 6 reach test.

Every policy is trained the same way (DAgger, the same seeds, rounds and action noise). Only
the thing between the 8 signals and the action differs:

    connectome   8 signals -> encoder -> male-CNS network -> 68 DNs -> linear readout
    random       8 signals -> encoder -> rewired network  -> 68 DNs -> linear readout
    MLP          8 signals (noise-free)                   -> small MLP (2 x 64, tanh)
    linear       8 signals (noise-free)                   -> linear readout   (reference)

The random network is shuffle_sub of the connectome: the post-synaptic end of every connection
is permuted across all connections. Every neuron keeps its number of inputs and outputs (so the
size and the in- and out-degree distributions are identical), every connection keeps its synapse
count and sign, and sign follows the presynaptic neuron (so the excitatory/inhibitory ratio of
neurons, connections and synapses is identical). Only who-connects-to-whom changes.

Both networks get the same rules, applied to their own wiring and activity:
  * encoder channels: the week 5 'wiring' split, computed on that network's own connections;
  * w_scale: the connectome keeps its week 2 value; the random network's is set so the same
    share of non-input neurons fires as in the connectome (the week 5 rule);
  * readout neurons: the 68 descending neurons most correlated with the expert's actions on
    the round 0 (expert-driven) data. Chosen once, after round 0, and then fixed.
"""
import time
import numpy as np

from .closed_loop import RidgeReadout, wilson

# ---------------------------------------------------------------- the 8 signals
SIGNALS8 = ('cube_rel_x', 'cube_rel_y', 'cube_rel_z', 'grip_width',
            'goal_rel_x', 'goal_rel_y', 'tcp_z', 'cube_z')
# (kind, centre, scale): s = tanh((x - centre) / scale), or (x - centre) / scale clipped for 'lin'.
# The first four are week 5's, unchanged. goal - cube uses the same 5 cm scale as cube - gripper,
# which also roughly matches where the expert's carry speed saturates (0.6 x error > 2 cm/step
# beyond 3.3 cm). Heights are centred between grasp height (3 cm) and travel height (13 cm).
SPEC8 = dict(cube_rel_x=('tanh', 0.0, 0.05), cube_rel_y=('tanh', 0.0, 0.05),
             cube_rel_z=('tanh', 0.0, 0.08), grip_width=('lin', 0.04, 0.04),
             goal_rel_x=('tanh', 0.0, 0.05), goal_rel_y=('tanh', 0.0, 0.05),
             tcp_z=('tanh', 0.08, 0.05), cube_z=('tanh', 0.05, 0.04))
ACT4 = ('dx', 'dy', 'dz', 'grip')


def normalise8(obs, obs_names, signals=SIGNALS8):
    """Observations (n, 24) -> s (n, len(signals)) in [-1, 1]. Matches week 5's normalise()
    on the first 4 signals."""
    obs = np.atleast_2d(np.asarray(obs, float))
    col = {n: i for i, n in enumerate(list(obs_names))}
    out = np.empty((len(obs), len(signals)))
    for k, name in enumerate(signals):
        kind, c, sc = SPEC8[name]
        x = (obs[:, col[name]] - c) / sc
        out[:, k] = np.clip(x, -1, 1) if kind == 'lin' else np.tanh(x)
    return out


# ---------------------------------------------------------------- task constants (fixed before any brain run)
PP_MAX_STEPS = 150      # 6 s. Expert: median 49 steps clean, 58 with 5 mm noise; 104 at 10 mm (week 4)
PP_HOLD = 10            # success must hold 10 steps (0.4 s), as in week 4's rollout.py
LIFT_Z = 0.045          # m, cube centre this high (2 cm off the table) counts as lifted
OFF_TABLE_Z = -0.01     # m, cube fell off the table: stop the episode
STALL_STEPS = 25        # 1 s with the gripper and cube each moving < STALL_MM, and no success: stop.
STALL_MM = 5.0          # A static readout can settle where its motion output cancels to zero
                        # (seen in the sandbox: hovering over the zone holding the cube until
                        # timeout). The expert never stands still this long; stopping saves brain time.
ZONE_R = 0.05           # m, panda_env.ZONE_RADIUS
NO_LIFT_STEPS = 100     # 4 s without lifting the cube: stop as 'never lifted'. The expert lifts at
                        # about 0.9 s and needs about 1.2 s from lift to done, so a policy lifting
                        # at 4 s could still finish inside the 6 s limit. Spiking noise keeps the
                        # gripper jittering, so the stall stop rarely fires for a brain; without this,
                        # every brain episode that never grasps would cost the full 6 s.


# ---------------------------------------------------------------- readouts
class RidgeMulti:
    """Week 6's RidgeReadout (episode-grouped CV over the same alphas, standardised features),
    but with one penalty per GROUP of target columns, sharing one SVD per fold:
    groups=((0, 1, 2), (3,)) gives motion and grip their own alpha at the cost of one fit."""

    def __init__(self, col_groups=((0, 1, 2), (3,)), alphas=None, n_folds=5, seed=0):
        from .readout import ALPHAS
        self.col_groups = [list(c) for c in col_groups]
        self.alphas = ALPHAS if alphas is None else alphas
        self.n_folds, self.seed = n_folds, seed

    @staticmethod
    def _svd(Z, Yc):
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        return Vt, s, U.T @ Yc

    def _coef(self, Vt, s, UtY, a):
        return Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY)

    def fit(self, X, Y, groups):
        from .readout import _folds
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        mu, sd = X.mean(0), X.std(0)
        self.ok = sd > 1e-9
        self.mu, self.sd = mu[self.ok], sd[self.ok]
        err = np.zeros((len(self.col_groups), len(self.alphas)))
        if len(np.unique(groups)) >= self.n_folds:
            for f in _folds(groups, self.n_folds, self.seed):
                Xa, Xb = X[~f], X[f]
                m_, s_ = Xa.mean(0), Xa.std(0); ok = s_ > 1e-9
                Za, Zb = (Xa[:, ok] - m_[ok]) / s_[ok], (Xb[:, ok] - m_[ok]) / s_[ok]
                ym = Y[~f].mean(0)
                Vt, s, UtY = self._svd(Za, Y[~f] - ym)
                for i, a in enumerate(self.alphas):
                    e = (Zb @ self._coef(Vt, s, UtY, a) + ym - Y[f]) ** 2
                    for gi, cols in enumerate(self.col_groups):
                        err[gi, i] += e[:, cols].sum()
            self.alpha = [float(self.alphas[int(np.argmin(e))]) for e in err]
        else:
            self.alpha = [float(self.alphas[len(self.alphas) // 2])] * len(self.col_groups)
        Z = (X[:, self.ok] - self.mu) / self.sd
        self.ym = Y.mean(0)
        Vt, s, UtY = self._svd(Z, Y - self.ym)
        self.W = np.zeros((Z.shape[1], Y.shape[1]))
        for a, cols in zip(self.alpha, self.col_groups):
            self.W[:, cols] = self._coef(Vt, s, UtY[:, cols], a)
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return ((X[:, self.ok] - self.mu) / self.sd) @ self.W + self.ym


class ActionReadout:
    """features -> (dx, dy, dz, grip). Ridge (default; motion and grip each get their own
    penalty) or a small MLP (standardised targets, so the loss weighs all four equally)."""

    def __init__(self, kind='ridge', **kw):
        self.kind, self.kw = kind, kw

    def fit(self, X, Y, groups):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if self.kind == 'ridge':
            self.lin = RidgeMulti(**self.kw).fit(X, Y, groups)
            self.alpha = tuple(self.lin.alpha)
        else:
            self.net = MLP(**self.kw).fit(X, Y, groups)
            self.alpha = None
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if self.kind == 'ridge':
            return self.lin.predict(X)
        return self.net.predict(X)


class MLP:
    """Small MLP policy in plain numpy (no torch needed in sim-env): standardised inputs,
    2 hidden tanh layers, standardised targets, Adam, weight decay. The output columns are
    standardised, so the loss weighs dx, dy, dz and grip equally."""

    def __init__(self, hidden=(64, 64), epochs=300, batch=256, lr=3e-3, wd=1e-4, seed=0):
        self.hidden, self.epochs, self.batch, self.lr, self.wd, self.seed = hidden, epochs, batch, lr, wd, seed

    def fit(self, X, Y, groups=None):
        rng = np.random.default_rng(self.seed)
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-9
        self.my, self.sy = Y.mean(0), Y.std(0) + 1e-9
        Xs, Ys = (X - self.mx) / self.sx, (Y - self.my) / self.sy
        sizes = [X.shape[1], *self.hidden, Y.shape[1]]
        self.P = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.P += [rng.normal(0, np.sqrt(1 / a), (a, b)), np.zeros(b)]
        m = [np.zeros_like(p) for p in self.P]; v = [np.zeros_like(p) for p in self.P]
        b1, b2, step = 0.9, 0.999, 0
        n = len(Xs)
        for ep in range(self.epochs):
            perm = rng.permutation(n)
            lr = self.lr * 0.5 * (1 + np.cos(np.pi * ep / self.epochs))        # cosine decay
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                grads = self._grads(Xs[idx], Ys[idx])
                step += 1
                for j, (p, g) in enumerate(zip(self.P, grads)):
                    if j % 2 == 0:
                        g = g + self.wd * p
                    m[j] = b1 * m[j] + (1 - b1) * g
                    v[j] = b2 * v[j] + (1 - b2) * g * g
                    p -= lr * (m[j] / (1 - b1 ** step)) / (np.sqrt(v[j] / (1 - b2 ** step)) + 1e-8)
        return self

    def _forward(self, X):
        hs = [X]
        for j in range(0, len(self.P) - 2, 2):
            hs.append(np.tanh(hs[-1] @ self.P[j] + self.P[j + 1]))
        return hs, hs[-1] @ self.P[-2] + self.P[-1]

    def _grads(self, X, Y):
        hs, out = self._forward(X)
        d = 2 * (out - Y) / len(X)
        grads = [None] * len(self.P)
        grads[-2], grads[-1] = hs[-1].T @ d, d.sum(0)
        for j in range(len(self.P) - 4, -1, -2):
            d = (d @ self.P[j + 2].T) * (1 - hs[j // 2 + 1] ** 2)
            grads[j], grads[j + 1] = hs[j // 2].T @ d, d.sum(0)
        return grads

    def predict(self, X):
        return self._forward((np.atleast_2d(X) - self.mx) / self.sx)[1] * self.sy + self.my


# ---------------------------------------------------------------- policies
class ExpertPolicy:
    name = 'expert'

    def __init__(self, expert, obs_names, signals=SIGNALS8):
        self.expert, self.obs_names, self.signals = expert, obs_names, signals
        self.counts = None

    def reset(self, seed, obs):
        pass

    def act(self, obs):
        self.s = normalise8(obs, self.obs_names, self.signals)[0]
        return np.asarray(self.expert(obs)[0], float), self.s


class SignalPolicy:
    """No brain: a readout (linear or MLP) applied straight to the 8 noise-free signals."""

    def __init__(self, obs_names, kind='ridge', name=None, signals=SIGNALS8, **kw):
        self.obs_names, self.kind, self.signals, self.kw = obs_names, kind, signals, kw
        self.name = name or ('MLP' if kind == 'mlp' else 'linear')
        self.readout, self.counts = None, None

    def reset(self, seed, obs):
        pass

    def features(self, obs):
        self.s = normalise8(obs, self.obs_names, self.signals)[0]
        return self.s

    def act(self, obs):
        x = self.features(obs)
        a = np.r_[0, 0, 0, 1.0] if self.readout is None else self.readout.predict(x)[0]
        return a, x

    def fit(self, X, Y, g, rounds=None):
        self.readout = ActionReadout(self.kind, **self.kw).fit(X, Y, g)
        return self.readout


def informative(X, Y, k):
    """Indices of the k columns of X most correlated with any target column (max |Pearson r|).
    Silent or constant columns score 0."""
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    sx = np.sqrt((Xc ** 2).sum(0)); sy = np.sqrt((Yc ** 2).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (Xc.T @ Yc) / np.outer(sx, sy)
    score = np.nan_to_num(np.abs(r)).max(1)
    order = np.argsort(-score, kind='stable')
    return np.sort(order[:k]), score


class BrainPolicy:
    """A spiking network drives the arm. act(obs) advances the brain brain_ms and returns
    (dx, dy, dz, grip).

    features(): rates (Hz) over the last window_ms of EVERY recorded neuron (BrainIO.record),
    so any readout pool can be refit afterwards without rerunning the brain.
    pool: columns the readout may choose from; k: how many it keeps (chosen by `informative`
    on the data available at the first fit, i.e. round 0, then fixed).
    keep_bins: also keep each step's 5 ms bins of the pool (for the window analysis)."""

    def __init__(self, io, w_scale, obs_names, pool, k=68, brain_ms=40, window_ms=40, warmup_ms=200,
                 name='brain', signals=SIGNALS8, keep_bins=False):
        from .interface import Decoder
        self.io, self.w, self.obs_names, self.name, self.signals = io, w_scale, obs_names, name, signals
        n = len(io.record)
        assert brain_ms >= window_ms, "the decoder window must fit in one brain advance"
        self.dec = Decoder(np.arange(n), window_ms, io.bin_ms)
        self.pool, self.k = np.asarray(pool), k
        self.cols = None
        self.brain_ms, self.window_ms, self.warmup_ms = brain_ms, window_ms, warmup_ms
        self.readout, self.counts, self.keep_bins = None, None, keep_bins
        self.feedback = True

    def reset(self, seed, obs):
        self.io.reset(seed, self.w)
        self.s0 = normalise8(obs, self.obs_names, self.signals)[0]
        self.io.step(self.s0, self.warmup_ms)

    def features(self, obs):
        s = normalise8(obs, self.obs_names, self.signals)[0] if self.feedback else self.s0
        self.counts, _ = self.io.step(s, self.brain_ms)
        self.s = s
        return self.dec.rates(self.counts)

    def act(self, obs):
        x = self.features(obs)
        if self.readout is None:
            a = np.r_[0, 0, 0, 1.0]
        else:
            a = self.readout.predict(x[self.cols])[0]
        if self.keep_bins:
            self.last_bins = self.counts[:, self.pool].copy()
        return a, x

    def fit(self, X, Y, g, rounds=None):
        if self.cols is None:
            if self.k is None or self.k >= len(self.pool):
                self.cols = self.pool                      # the whole pool, no selection
            else:
                sel, self.score = informative(X[:, self.pool], Y, self.k)
                self.cols = self.pool[sel]
        self.readout = ActionReadout('ridge').fit(X[:, self.cols], Y, g)
        # raster for the video: the 100 readout neurons with the largest weights, grouped by output
        Wm = np.zeros((len(self.cols), 4))
        Wm[self.readout.lin.ok] = self.readout.lin.W / self.readout.lin.W.std(0).clip(1e-12)
        top = np.argsort(-np.abs(Wm).max(1))[:100]
        top = top[np.lexsort((-np.abs(Wm[top]).max(1), np.abs(Wm[top]).argmax(1)))]
        self.raster_cols = self.cols[top]
        return self.readout


class BatchBrainPolicy(BrainPolicy):
    """BrainPolicy for many episodes side by side (sim/batch7.BatchBrain). Same features, readout,
    feature selection and fitting; act() takes a list of observations and returns (B, 4) actions
    and (B, n_record) rates. Episodes that end are dropped from the batch (keep())."""
    batched = True

    def __init__(self, brain, w_scale, obs_names, pool, k=None, brain_ms=40, window_ms=40, warmup_ms=200,
                 name='brain', signals=SIGNALS8, keep_bins=False, max_batch=64):
        self.io, self.w, self.obs_names, self.name, self.signals = brain, w_scale, obs_names, name, signals
        assert brain_ms >= window_ms, "the decoder window must fit in one brain advance"
        self.nb_win = int(round(window_ms / brain.bin_ms))
        self.pool, self.k = np.asarray(pool), k
        self.cols, self.raster_cols = None, None
        self.brain_ms, self.window_ms, self.warmup_ms = brain_ms, window_ms, warmup_ms
        self.readout, self.counts, self.keep_bins = None, None, keep_bins
        self.feedback, self.max_batch = True, max_batch

    def reset(self, seeds, obs_list):
        self.io.reset(seeds, [self.w] * len(seeds))
        self.S0 = normalise8(np.array(obs_list), self.obs_names, self.signals)
        self.io.step(self.S0, self.warmup_ms)

    def act(self, obs_list):
        S = normalise8(np.array(obs_list), self.obs_names, self.signals) if self.feedback else self.S0
        self.S = S
        self.counts = self.io.step(S, self.brain_ms)                       # (B, n_bins, n_record)
        X = self.counts[:, -self.nb_win:, :].sum(1).astype(float) * (1000.0 / self.window_ms)
        if self.readout is None:
            A = np.tile(np.r_[0, 0, 0, 1.0], (len(X), 1))
        else:
            # row by row: a batched matrix product sums in a different order than a single row
            # (last-bit differences), which would make an episode depend on its batch
            A = np.vstack([self.readout.predict(x[None]) for x in X[:, self.cols]])
        if self.keep_bins:
            self.last_bins = self.counts[:, :, self.pool].copy()
        return A, X

    def keep(self, idx):
        self.io.keep(idx)
        self.S0 = self.S0[np.asarray(idx)]


# ---------------------------------------------------------------- one episode
class Episode:
    """One closed-loop pick-and-place episode, advanced one control step at a time, so that many
    can run side by side (one per environment) while a batched brain drives them all.
    Logs, for every visited state, the policy's features and the expert's label (the DAgger
    data). Success is env.info() (week 4 physics test) held PP_HOLD steps."""

    def __init__(self, env, expert, seed, expert_mix=0.0, action_noise=0.0, max_steps=PP_MAX_STEPS,
                 record=False):
        self.env, self.expert, self.seed = env, expert, seed
        self.mix, self.noise, self.max_steps, self.record = expert_mix, action_noise, max_steps, record
        self.rng = np.random.default_rng(seed + 7_000_000)
        self.obs = env.reset(seed)
        self.X, self.Y, self.A, self.O, self.PH, self.REC, self.BINS = [], [], [], [], [], [], []
        self.held, self.done_at, self.lifted_at, self.zone_at, self.flips, self.prev_grip = 0, None, None, None, 0, 1.0
        self.max_cube_z, self.stalled, self.t, self.done = 0.0, False, 0, False
        self.cz = float(self.obs[10])
        self.tcp_hist, self.cube_hist = [], []
        self.t0 = time.perf_counter()

    def step(self, a_pol, x, s=None, raster=None, bins=None):
        """Apply the policy's action for one control step. Returns True when the episode is over."""
        env, obs, t = self.env, self.obs, self.t
        label, phase = self.expert(obs)
        label = np.asarray(label, float)
        a = label.copy() if self.rng.random() < self.mix else np.asarray(a_pol, float)
        ex = a.copy()
        if self.noise:
            ex[:3] += self.rng.normal(0, self.noise, 3)
        g_cmd = 1.0 if ex[3] > 0 else -1.0
        self.flips += int(g_cmd != self.prev_grip); self.prev_grip = g_cmd
        self.X.append(np.asarray(x, np.float32)); self.Y.append(label); self.A.append(a)
        self.O.append(obs); self.PH.append(phase)
        if bins is not None:
            self.BINS.append(bins)
        if self.record:
            self.REC.append(dict(qpos=env.d.qpos.copy(), counts=None if raster is None else raster.copy(),
                                 s=None if s is None else np.array(s, copy=True), act=a.copy(),
                                 label=label.copy(), phase=phase))
        obs = self.obs = env.step(ex)
        cz = self.cz = float(obs[10])
        self.max_cube_z = max(self.max_cube_z, cz)
        if self.lifted_at is None and cz > LIFT_Z:
            self.lifted_at = t + 1
        if self.zone_at is None and cz > LIFT_Z and np.hypot(*obs[19:21]) < ZONE_R:
            self.zone_at = t + 1                      # carried the lifted cube over the zone
        info = env.info()
        self.held = self.held + 1 if info['success'] else 0
        if self.held == 1:
            self.done_at = t + 1
        if self.held == 0:
            self.done_at = None
        self.t = t + 1
        done = False
        if self.held >= PP_HOLD or cz < OFF_TABLE_Z:
            done = True
        elif self.lifted_at is None and t + 1 >= NO_LIFT_STEPS:
            done = True
        else:
            self.tcp_hist.append(obs[:3].copy()); self.cube_hist.append(obs[8:11].copy())
            if self.held == 0 and len(self.tcp_hist) > STALL_STEPS:
                moved = 1000 * max(np.linalg.norm(np.ptp(np.array(self.tcp_hist[-STALL_STEPS:]), 0)),
                                   np.linalg.norm(np.ptp(np.array(self.cube_hist[-STALL_STEPS:]), 0)))
                if moved < STALL_MM:
                    self.stalled = done = True
        if self.t >= self.max_steps:
            done = True
        if done:
            self.done = True
            if self.record:
                self.REC.append(dict(qpos=env.d.qpos.copy(), counts=None, s=None, act=None, label=None, phase=None))
        return done

    def result(self):
        info = self.env.info()
        cz = self.cz
        success = self.held >= PP_HOLD
        if success:
            outcome = 'success'
        elif cz < OFF_TABLE_Z:
            outcome = 'cube off table'
        elif self.lifted_at is None:
            outcome = 'never lifted'
        elif cz > LIFT_Z:
            outcome = 'still holding the cube'      # carrying, or hovering with it
        elif info['in_zone']:
            outcome = 'put down in zone, not finished'   # not released, gripper not clear, or not at rest
        else:
            outcome = 'dropped or put down outside the zone'
        return dict(seed=self.seed, success=success, outcome=outcome, stalled=self.stalled,
                    steps=self.done_at if success else None,
                    lifted=self.lifted_at is not None, lifted_steps=self.lifted_at,
                    zone=self.zone_at is not None, zone_steps=self.zone_at,
                    place_error_mm=1000 * info['place_error'], in_zone=info['in_zone'],
                    grip_switches=self.flips, n=len(self.X), wall_s=time.perf_counter() - self.t0,
                    X=np.array(self.X), Y=np.array(self.Y), A=np.array(self.A), obs=np.array(self.O),
                    phase=np.array(self.PH), bins=np.array(self.BINS) if self.BINS else None,
                    rec=self.REC if self.record else None)


def run_pp(env, policy, expert, seed, expert_mix=0.0, action_noise=0.0, max_steps=PP_MAX_STEPS,
           record=False):
    """One episode with a one-at-a-time policy (expert, linear, MLP, or week 6-style BrainPolicy)."""
    ep = Episode(env, expert, seed, expert_mix, action_noise, max_steps, record)
    policy.reset(seed, ep.obs)
    while not ep.done:
        a_pol, x = policy.act(ep.obs)
        raster = None
        if record:
            rc = getattr(policy, 'raster_cols', None)
            if getattr(policy, 'counts', None) is not None and rc is not None:
                raster = policy.counts[:, rc]
        ep.step(a_pol, x, s=getattr(policy, 's', None), raster=raster,
                bins=policy.last_bins if getattr(policy, 'keep_bins', False) else None)
    return ep.result()


class EnvPool:
    """MuJoCo environments sharing one compiled model, one per episode running side by side."""

    def __init__(self, env_or_model=None):
        from panda_env import PandaPickPlace, build_model
        self._cls = PandaPickPlace
        if env_or_model is None:
            self.m, self.envs = build_model(), []
        elif isinstance(env_or_model, PandaPickPlace):
            self.m, self.envs = env_or_model.m, [env_or_model]
        else:
            self.m, self.envs = env_or_model, []

    def get(self, n):
        while len(self.envs) < n:
            self.envs.append(self._cls(self.m))
        return self.envs[:n]


def as_pool(env):
    return env if isinstance(env, EnvPool) else EnvPool(env)


def run_episodes(env, policy, expert, seeds, mixes=None, noises=None, max_steps=PP_MAX_STEPS,
                 record=False):
    """Run episodes for `seeds`. A batched policy (BatchBrainPolicy) runs up to policy.max_batch of
    them side by side; any other policy runs them one after another. Results in seed order."""
    pool = as_pool(env)
    n = len(seeds)
    mixes = [0.0] * n if mixes is None else list(mixes)
    noises = [0.0] * n if noises is None else list(noises)
    if not getattr(policy, 'batched', False):
        e0 = pool.get(1)[0]
        return [run_pp(e0, policy, expert, sd, mx, nz, max_steps, record)
                for sd, mx, nz in zip(seeds, mixes, noises)]
    out = []
    for c0 in range(0, n, policy.max_batch):
        sl = slice(c0, min(n, c0 + policy.max_batch))
        envs = pool.get(sl.stop - sl.start)
        eps = [Episode(e, expert, sd, mx, nz, max_steps, record)
               for e, sd, mx, nz in zip(envs, seeds[sl], mixes[sl], noises[sl])]
        policy.reset(list(seeds[sl]), [ep.obs for ep in eps])
        alive = list(range(len(eps)))
        while alive:
            A, X = policy.act([eps[i].obs for i in alive])
            keep = []
            for j, i in enumerate(alive):
                raster = policy.counts[j][:, policy.raster_cols] if record and policy.raster_cols is not None \
                    and policy.counts is not None else None
                bins = policy.last_bins[j] if policy.keep_bins else None
                if not eps[i].step(A[j], X[j], s=policy.S[j], raster=raster, bins=bins):
                    keep.append(j)
            if len(keep) < len(alive):
                if keep:
                    policy.keep(keep)
                alive = [alive[j] for j in keep]
        out += [ep.result() for ep in eps]
    return out


# ---------------------------------------------------------------- DAgger and evaluation
def dagger_pp(env, policy, expert, train_seeds, betas=(1.0, 0.0, 0.0, 0.0), noise=0.005,
              verbose=True, on_round=None, max_steps=PP_MAX_STEPS, resume=None):
    """Week 6's DAgger with 4 outputs. train_seeds: one seed list per round. In every round,
    half the episodes add `noise` (m per step) to the executed motion. After each round the
    policy is refit (policy.fit) on everything collected so far.
    resume: (data, log) from a checkpoint (data as saved by on_round: concatenated arrays);
    the finished rounds are skipped and the policy is refit on their data first. Episodes are
    seeded, so on the CPU a resumed run collects exactly what an uninterrupted one would have
    (on a GPU, rounding can differ slightly; see batch7.py)."""
    data = dict(X=[], Y=[], g=[], round=[], obs=[], bins=[])
    log = []
    if resume is not None:
        d0, log = resume
        log = list(log)
        for k in ('X', 'Y', 'g', 'round', 'obs'):
            data[k].append(np.asarray(d0[k]))
        X, Y, g = (np.concatenate(data[k]) for k in ('X', 'Y', 'g'))
        policy.fit(X, Y, g)
        if verbose:
            print(f"  resumed after round {len(log) - 1}: {len(X):,} rows")
    t0 = time.perf_counter()
    for r, (beta, seeds) in enumerate(zip(betas, train_seeds)):
        if r < len(log):
            continue
        outs = run_episodes(env, policy, expert, list(seeds), mixes=[beta] * len(seeds),
                            noises=[noise if i % 2 else 0.0 for i in range(len(seeds))], max_steps=max_steps)
        for sd, o in zip(seeds, outs):
            data['X'].append(o['X']); data['Y'].append(o['Y']); data['obs'].append(o['obs'])
            data['g'].append(np.full(o['n'], sd)); data['round'].append(np.full(o['n'], r))
            if o['bins'] is not None:
                data['bins'].append(o['bins'])
        X, Y, g = (np.concatenate(data[k]) for k in ('X', 'Y', 'g'))
        policy.fit(X, Y, g)
        row = dict(round=r, beta=beta, episodes=len(seeds), rows=len(X),
                   alpha=getattr(policy.readout, 'alpha', None),
                   success=float(np.mean([o['success'] for o in outs])),
                   lifted=float(np.mean([o['lifted'] for o in outs])),
                   wall_s=time.perf_counter() - t0)
        log.append(row)
        if verbose:
            who = 'expert drives' if beta == 1 else f'{policy.name} drives'
            print(f"  round {r} ({who}): {len(seeds)} episodes, success while collecting "
                  f"{row['success']:.0%}, lifted {row['lifted']:.0%}; refit on {len(X):,} rows  "
                  f"[{row['wall_s']:.0f} s]")
        if on_round is not None:
            on_round(r, data, log)
    for k in ('X', 'Y', 'g', 'round', 'obs'):
        data[k] = np.concatenate(data[k])
    data['bins'] = np.concatenate(data['bins']) if data['bins'] else None
    return data, log


def evaluate_pp(env, policy, expert, seeds, verbose=True, label='', keep=False, record=False):
    t0 = time.perf_counter()
    outs = run_episodes(env, policy, expert, list(seeds), record=record)
    if not keep:
        outs = [{k: v for k, v in o.items() if k not in ('X', 'bins')} for o in outs]   # drop the big arrays
    s = summarise_pp(outs)
    if verbose:
        print(f"  {label or policy.name}: success {s['success']}/{s['n']} ({s['success_rate']:.0%}, "
              f"95% CI {s['ci'][0]:.0%}-{s['ci'][1]:.0%}), placement {s['place_mm_success']} mm, "
              f"time {s['time_s']} s, lifted {s['lifted_rate']:.0%}  [{time.perf_counter() - t0:.0f} s]")
    return outs, s


def summarise_pp(outs):
    """The three deliverable metrics, plus the failure breakdown.
      success rate          week 4 physics test held 0.4 s within 6 s
      placement accuracy    cube-to-target distance at the end (mm): median over successes,
                            and median over every episode in which the cube was lifted
      time to complete      first step of the successful hold, in s (median over successes)"""
    n = len(outs)
    k = sum(o['success'] for o in outs)
    succ = [o for o in outs if o['success']]
    lifted = [o for o in outs if o['lifted']]
    med = lambda v: float(np.median(v)) if len(v) else None
    outcomes = {}
    for o in outs:
        outcomes[o['outcome']] = outcomes.get(o['outcome'], 0) + 1
    zone = [o for o in outs if o['zone']]
    return dict(n=n, success=int(k), success_rate=k / n, ci=wilson(k, n),
                place_mm_success=None if not succ else round(med([o['place_error_mm'] for o in succ]), 1),
                place_mm_lifted=None if not lifted else round(med([o['place_error_mm'] for o in lifted]), 1),
                time_s=None if not succ else round(0.04 * med([o['steps'] for o in succ]), 2),
                lifted_rate=len(lifted) / n, zone_rate=len(zone) / n,
                lift_time_s=None if not lifted else round(0.04 * med([o['lifted_steps'] for o in lifted]), 2),
                stalled_rate=float(np.mean([o.get('stalled', False) for o in outs])),
                outcomes=outcomes, grip_switches=med([o['grip_switches'] for o in outs]))


def paired_milestones(outs_a, outs_b):
    """Paired comparison of the progress milestones (lifted, carried to the zone, success) on
    the same seeds: how many seeds only a reached, only b reached, and an exact two-sided sign
    (McNemar) test p-value on those discordant seeds."""
    from math import comb
    A = {o['seed']: o for o in outs_a}; B = {o['seed']: o for o in outs_b}
    seeds = sorted(set(A) & set(B))
    out = {}
    for m in ('lifted', 'zone', 'success'):
        a = np.array([A[s][m] for s in seeds]); b = np.array([B[s][m] for s in seeds])
        na, nb = int((a & ~b).sum()), int((~a & b).sum())
        nd = na + nb
        p = 1.0 if nd == 0 else min(1.0, 2 * sum(comb(nd, i) for i in range(0, min(na, nb) + 1)) / 2 ** nd)
        out[m] = dict(a=float(a.mean()), b=float(b.mean()), only_a=na, only_b=nb, p=p)
    return out


def paired_diff(outs_a, outs_b, n_boot=10_000, seed=0):
    """Success-rate difference a - b on the SAME seeds, with a paired bootstrap 95% CI
    (resampling seeds), and the counts of seeds only a / only b solved."""
    sa = {o['seed']: o['success'] for o in outs_a}
    sb = {o['seed']: o['success'] for o in outs_b}
    seeds = sorted(set(sa) & set(sb))
    a = np.array([sa[s] for s in seeds], float); b = np.array([sb[s] for s in seeds], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(seeds), (n_boot, len(seeds)))
    d = (a[idx] - b[idx]).mean(1)
    return dict(diff=float(a.mean() - b.mean()), ci=(float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
                only_a=int(((a == 1) & (b == 0)).sum()), only_b=int(((a == 0) & (b == 1)).sum()), n=len(seeds))


# ---------------------------------------------------------------- timing: brain ms per control step, decoder window
def timing_sweep(env, make_policy, expert, seeds, settings=((40, (20, 40)), (80, (40, 80))), verbose=True):
    """Pick brain time per control step and decoder window by one rule, for any network.

    For each brain_ms, the expert drives `seeds` (half with 5 mm action noise, as DAgger round 0)
    while the network watches; the 5 ms spike bins of the readout pool are kept, so every window
    up to brain_ms is scored from the same run. Score per setting, all out-of-episode (5 folds by
    episode, ridge): mean of the dx, dy, dz R2 and of grip skill (2 x balanced accuracy - 1).
    Also reported: the same score with the features one control step late (lag 1; if that is
    higher, the network reports the state late), and wall time per control step (stability).

    make_policy(brain_ms) -> BrainPolicy (window irrelevant here)."""
    from .readout import cv_r2, cv_predict
    rows, keep = [], {}
    for brain_ms, windows in settings:
        pol = make_policy(brain_ms)
        pol.keep_bins = True
        t0 = time.perf_counter()
        data, _ = dagger_pp(env, pol, expert, [list(seeds)], betas=[1.0], verbose=False)
        wall = time.perf_counter() - t0
        B, Y, g = data['bins'], data['Y'], data['g']
        first = np.r_[True, g[1:] != g[:-1]]                  # episodes are stored contiguously
        step = np.arange(len(g)) - np.maximum.accumulate(np.where(first, np.arange(len(g)), 0))
        keep[brain_ms] = (B, Y, g)
        for w in windows:
            nb = int(round(w / 5))
            X = B[:, -nb:, :].sum(1).astype(float) * (1000.0 / w)
            for lag in (0, 1):
                if lag:
                    m = step >= 1
                    Xl, Yl, gl = X[np.flatnonzero(m) - 1], Y[m], g[m]
                else:
                    Xl, Yl, gl = X, Y, g
                r2m = cv_r2(Xl, Yl[:, :3], gl)
                pg = cv_predict(Xl, Yl[:, 3], gl)[0]
                lab = Yl[:, 3] > 0
                bal = 0.5 * (np.mean((pg > 0)[lab]) + np.mean((pg <= 0)[~lab]))
                rows.append(dict(brain_ms=brain_ms, window_ms=w, lag=lag, dx=r2m[0], dy=r2m[1], dz=r2m[2],
                                 grip_bal_acc=bal, score=float(np.mean([*r2m, 2 * bal - 1])),
                                 wall_ms_per_step=1000 * wall / len(Y), steps=len(Y)))
        if verbose:
            print(f"  brain {brain_ms} ms per control step: {len(Y)} steps, "
                  f"{1000 * wall / len(Y):.0f} ms wall per step")
    return rows, keep


def choose_timing(rows, cost_margin=0.05, tie=0.01, default=(40, 40)):
    """Rule fixed in advance: the highest lag-0 score wins, except that a setting with more brain
    time per step than the default (it costs proportionally more laptop time) must beat the best
    default-cost setting by cost_margin; and anything within `tie` of the default keeps the default."""
    r0 = [r for r in rows if r['lag'] == 0]
    sc = {(r['brain_ms'], r['window_ms']): r['score'] for r in r0}
    cheap = {k: v for k, v in sc.items() if k[0] <= default[0]}
    best_cheap = max(cheap, key=cheap.get)
    best = max(sc, key=sc.get)
    if best[0] > default[0] and sc[best] < sc[best_cheap] + cost_margin:
        best = best_cheap
    if default in sc and sc[best] < sc[default] + tie:
        best = default
    return best
