"""Week 5: the interface between the arm and the brain.

    task state --Encoder--> Poisson rate on each input neuron (JO-C/E)
               --BrainIO--> brain advanced one control step (40 ms) at a time,
                            spike counts of chosen neurons kept in 5 ms bins
               --Decoder--> firing rate of each output neuron over the last 20-50 ms

The same BrainIO.step() call is what the week 6 closed loop will use, so the replay in
week 5 exercises exactly the code path the closed loop will run.

Encoder design (opponent code, constant total drive):
  * Each scalar signal s in [-1, 1] owns two input channels, "push" and "pull".
    push rate = r_mid + r_amp * s,  pull rate = r_mid - r_amp * s  (25..275 Hz by default).
  * Every channel's pair sums to 2 * r_mid, so the total input rate never changes with the
    task state. The brain cannot carry information just by firing more or less overall;
    it has to route WHICH neurons are driven. That makes "outputs vary with input" a real test.
  * r_mid = 150 Hz is the Week 2/3 operating point, where w_syn was tuned.
  * Positions are squashed with tanh(rel / scale) so resolution is highest near the cube,
    where grasping needs it (scale 5 cm in x, y; 8 cm in z so the 13 cm hover height
    is not saturated). Gripper width 0..8 cm maps linearly to -1..1.

Channel assignment (which JO-C/E neurons form which channel):
  * 'wiring': JO neurons are grouped by what they connect to. Each neuron's row of direct
    output synapses is split recursively along its first principal component into
    2 x n_signals equal groups. The first split separates push from pull, so the two halves
    of every signal reach the most different downstream partners.
  * 'random': a seeded random equal split (the control).
"""
import json, time

import numpy as np
import pandas as pd

SIGNALS = ('cube_rel_x', 'cube_rel_y', 'cube_rel_z', 'grip_width')
SCALES = dict(cube_rel_x=0.05, cube_rel_y=0.05, cube_rel_z=0.08)   # m, tanh squashing
GRIP_MAX = 0.08                                                    # m, fully open


# ---------------------------------------------------------------- task state -> s
def normalise(obs, obs_names, signals=SIGNALS):
    """Logged observations (n, 24) -> encoded signals s (n, n_signals), each in [-1, 1]."""
    obs = np.atleast_2d(np.asarray(obs, float))
    col = {n: i for i, n in enumerate(list(obs_names))}
    out = np.empty((len(obs), len(signals)))
    for k, name in enumerate(signals):
        x = obs[:, col[name]]
        if name == 'grip_width':
            out[:, k] = np.clip(2 * x / GRIP_MAX - 1, -1, 1)
        else:
            out[:, k] = np.tanh(x / SCALES[name])
    return out


def denormalise(s, signals=SIGNALS):
    """Inverse of normalise (clipped so tanh^-1 stays finite). Returns metres."""
    s = np.atleast_2d(s)
    out = np.empty_like(s, dtype=float)
    for k, name in enumerate(signals):
        if name == 'grip_width':
            out[:, k] = (s[:, k] + 1) * GRIP_MAX / 2
        else:
            out[:, k] = np.arctanh(np.clip(s[:, k], -0.99, 0.99)) * SCALES[name]
    return out


# ---------------------------------------------------------------- channel assignment
def random_channels(n, n_channels, seed=0):
    """Seeded random equal split of n neurons into n_channels groups."""
    lab = np.empty(n, int)
    for c, part in enumerate(np.array_split(np.random.default_rng(seed).permutation(n), n_channels)):
        lab[part] = c
    return lab


def wiring_channels(ws, neurons, n_channels):
    """Group input neurons by their direct output connectivity (see module docstring).
    n_channels must be a power of two. Returns a label per neuron (0..n_channels-1);
    labels < n_channels/2 lie on one side of the first split."""
    depth = int(round(np.log2(n_channels)))
    assert 2 ** depth == n_channels, "n_channels must be a power of two"
    A = ws._adj()[np.asarray(neurons)]               # rows: input neurons, cols: all neurons
    cols = np.unique(A.indices)
    P = A[:, cols].toarray().astype(float)
    P = np.log1p(P)                                  # many small vs a few large counts
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12

    def split(idx, d):
        if d == 0:
            return [idx]
        X = P[idx] - P[idx].mean(0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        score = X @ vt[0]
        if vt[0][np.argmax(np.abs(vt[0]))] < 0:      # fix the sign so results are reproducible
            score = -score
        order = idx[np.argsort(score, kind='stable')]
        lo, hi = np.array_split(order, 2)
        return split(lo, d - 1) + split(hi, d - 1)

    lab = np.empty(len(neurons), int)
    for c, part in enumerate(split(np.arange(len(neurons)), depth)):
        lab[part] = c
    return lab


def channel_overlap(ws, neurons, labels):
    """Fraction of output synapses that two channels send to the same partners
    (cosine similarity of their summed output rows). Low off-diagonal = channels
    reach different neurons."""
    A = ws._adj()[np.asarray(neurons)]
    k = labels.max() + 1
    rows = np.vstack([np.asarray(A[labels == c].sum(0)).ravel() for c in range(k)]).astype(float)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12
    return rows @ rows.T


# ---------------------------------------------------------------- encoder
class Encoder:
    """s (n_signals,) -> Poisson rate (Hz) for each input neuron, in the order of self.neurons.
    Channel c < K is the push channel of signal c; channel K + c is its pull channel."""

    def __init__(self, neurons, labels, signals=SIGNALS, r_mid=150.0, r_amp=125.0, kind=''):
        self.neurons = np.asarray(neurons)
        self.labels = np.asarray(labels)
        self.signals = tuple(signals)
        self.K = len(self.signals)
        assert self.labels.max() + 1 == 2 * self.K, "need 2 channels per signal"
        self.r_mid, self.r_amp, self.kind = float(r_mid), float(r_amp), kind
        self.sig_of = self.labels % self.K                       # which signal
        self.sgn = np.where(self.labels < self.K, 1.0, -1.0)     # push +, pull -

    def rates(self, s):
        s = np.clip(np.asarray(s, float), -1, 1)
        return self.r_mid + self.r_amp * self.sgn * s[self.sig_of]

    def to_json(self, path, ws=None):
        d = dict(kind=self.kind, signals=self.signals, r_mid=self.r_mid, r_amp=self.r_amp,
                 scales=SCALES, grip_max=GRIP_MAX, neurons=self.neurons.tolist(),
                 labels=self.labels.tolist())
        if ws is not None:
            d['bodyIds'] = ws.neurons.loc[self.neurons, 'bodyId'].astype(np.int64).tolist()
        open(path, 'w').write(json.dumps(d))

    @classmethod
    def from_json(cls, path):
        d = json.loads(open(path).read())
        return cls(d['neurons'], d['labels'], d['signals'], d['r_mid'], d['r_amp'], d['kind'])


# ---------------------------------------------------------------- brain stepping
class BrainIO:
    """Drives a NumbaLIF engine (sim/lif_numba.py) one control step at a time.

    record: global indices of neurons whose spikes are kept, in bin_ms bins.
    Only those are kept (plus a per-bin total and a per-neuron total over the whole run),
    so a long replay of the full network fits in a few hundred MB.
    """

    def __init__(self, engine, encoder, record, bin_ms=5, ctrl_ms=40):
        self.eng, self.enc = engine, encoder
        assert np.array_equal(np.asarray(engine.keep)[engine.stim], encoder.neurons), \
            "engine must be built with stim_global = encoder.neurons (same order)"
        self.record = np.asarray(record)
        g2l = pd.Series(np.arange(len(engine.keep)), index=engine.keep)
        self.lut = np.full(len(engine.keep), -1, np.int64)
        self.lut[g2l.loc[self.record].to_numpy()] = np.arange(len(self.record))
        self.bin_ms, self.ctrl_ms = bin_ms, ctrl_ms
        self.steps_per_bin = int(round(bin_ms / engine.dt))
        self.bins_per_ctrl = int(round(ctrl_ms / bin_ms))
        self.neuron_counts = np.zeros(len(engine.keep), np.int64)   # whole-run totals
        self.sim_ms = 0.0

    def reset(self, seed, w_scale):
        self.eng.reset(1, seed=seed, w_scale=w_scale)

    def advance(self, rates, n_ms):
        """Advance n_ms with fixed input rates. Returns (counts (bins, n_record), total per bin)."""
        nb = int(round(n_ms / self.bin_ms))
        fi, ft = self.eng.advance(nb * self.steps_per_bin, rate=rates)
        b = (ft - (self.eng.step_i - nb * self.steps_per_bin)) // self.steps_per_bin
        col = self.lut[fi]
        m = col >= 0
        counts = np.zeros((nb, len(self.record)), np.uint16)
        np.add.at(counts, (b[m], col[m]), 1)
        self.neuron_counts += np.bincount(fi, minlength=len(self.neuron_counts))
        self.sim_ms += n_ms
        return counts, np.bincount(b, minlength=nb)

    def step(self, s, n_ms=None):
        """One control step: encode s, advance ctrl_ms. This is the closed-loop call."""
        return self.advance(self.enc.rates(s), n_ms or self.ctrl_ms)


class Decoder:
    """Firing rates (Hz) of output neurons over the last window_ms.
    cols: columns of BrainIO.record that are this decoder's output neurons."""

    def __init__(self, cols, window_ms=40, bin_ms=5):
        self.cols = np.asarray(cols)
        self.window_ms, self.bin_ms = window_ms, bin_ms
        self.nb = int(round(window_ms / bin_ms))

    def rates(self, counts):
        """counts: recent bins (bins, n_record), newest last. Online use (week 6)."""
        return counts[-self.nb:, self.cols].sum(0) * (1000.0 / self.window_ms)

    def windows(self, counts, end_bins):
        """Offline: rate vector for the window ending at each bin index in end_bins
        (exclusive end). Uses a cumulative sum, so any window length is cheap."""
        cs = np.zeros((len(counts) + 1, len(self.cols)), np.int64)
        np.cumsum(counts[:, self.cols], axis=0, out=cs[1:])
        end_bins = np.asarray(end_bins)
        return (cs[end_bins] - cs[end_bins - self.nb]) * (1000.0 / self.window_ms)


# ---------------------------------------------------------------- experiments
def replay(io, episodes, w_scale, warmup_ms=200, seed0=0, verbose=True):
    """Replay logged episodes through the brain, one 40 ms control step per logged state.

    episodes: list of (episode_id, s) with s an (n_steps, n_signals) array.
    Each episode starts from a fresh brain that is first held at its first state for
    warmup_ms (as the closed loop will do), seed = seed0 + episode_id.
    Returns dict with the binned counts of all episodes concatenated, the bin index at
    which each control step ends, and per-step episode id and step number."""
    counts, totals, end_bin, ep_of, t_of = [], [], [], [], []
    offset, t0 = 0, time.perf_counter()
    for k, (ep, s) in enumerate(episodes):
        io.reset(seed0 + int(ep), w_scale)
        c, tot = io.step(s[0], warmup_ms)
        cs, ts = [c], [tot]
        for x in s:
            c, tot = io.step(x)
            cs.append(c); ts.append(tot)
        c = np.concatenate(cs); counts.append(c); totals.append(np.concatenate(ts))
        nw = int(round(warmup_ms / io.bin_ms))
        end_bin.append(offset + nw + io.bins_per_ctrl * np.arange(1, len(s) + 1))
        ep_of.append(np.full(len(s), ep)); t_of.append(np.arange(len(s)))
        offset += len(c)
        if verbose and (k % 10 == 9 or k == len(episodes) - 1):
            el = time.perf_counter() - t0
            print(f"  {k + 1}/{len(episodes)} episodes, {io.sim_ms / 1000:.1f} s brain time, "
                  f"{el:.0f} s wall ({el / (io.sim_ms / 1000):.2f} s per simulated second)")
    return dict(counts=np.concatenate(counts), totals=np.concatenate(totals),
                end_bin=np.concatenate(end_bin), episode=np.concatenate(ep_of),
                step=np.concatenate(t_of), bin_ms=io.bin_ms)


def tuning_sweep(io, w_scale, levels=np.linspace(-1, 1, 9), hold_ms=300, settle_ms=100,
                 warmup_ms=200, seed=0, verbose=True):
    """Vary one signal at a time (others held at 0), levels in random order, each held for
    hold_ms. Fresh brain per signal. Returns a long table: one row per 40 ms window in the
    settled part of each level, and the full binned counts for transient analysis."""
    rng = np.random.default_rng(seed)
    K = io.enc.K
    rows, trans = [], []
    nb_settle = int(round(settle_ms / io.bin_ms))
    nb_hold = int(round(hold_ms / io.bin_ms))
    t0 = time.perf_counter()
    for k in range(K):
        io.reset(seed * 1000 + k, w_scale)
        io.step(np.zeros(K), warmup_ms)
        prev = 0.0
        for lv in rng.permutation(levels):
            s = np.zeros(K); s[k] = lv
            c, tot = io.step(s, hold_ms)
            trans.append(dict(signal=k, level=lv, prev=prev, counts=c, totals=tot))
            prev = lv
            for w0 in range(nb_settle, nb_hold - io.bins_per_ctrl + 1, io.bins_per_ctrl):
                rows.append(dict(signal=k, level=lv, seed=seed,
                                 counts=c[w0:w0 + io.bins_per_ctrl].sum(0)))
    if verbose:
        print(f"  sweep seed {seed}: {time.perf_counter() - t0:.0f} s wall")
    return rows, trans
