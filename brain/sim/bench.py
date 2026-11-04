"""Week 3 helpers: Brian2 with fixed input spikes, spike-train comparison,
subnetwork extraction, readout metrics and timing."""
import time

import numpy as np
import pandas as pd
from brian2 import (NeuronGroup, Synapses, SpikeGeneratorGroup, SpikeMonitor, Network,
                    mV, ms, second, defaultclock)

from .lif import SHIU, EQS


# ---------------------------------------------------------------- shared inputs
def poisson_inputs(n_stim, rate_hz=150, t_run_s=1.0, n_trials=1, seed=0, dt_ms=0.1):
    """One bool (n_steps, n_stim) array of stimulus spikes per trial.
    Same statistics as Brian2's PoissonGroup (spike when rand < rate*dt)."""
    T = int(round(t_run_s * 1000 / dt_ms))
    out = []
    for k in range(n_trials):
        rng = np.random.default_rng(seed + k)
        out.append(rng.random((T, n_stim)) < rate_hz * dt_ms * 1e-3)
    return out


class B2Fixed:
    """The same Brian2 network as lif.LIFNet, but the stimulus comes from given
    spike arrays (SpikeGeneratorGroup) instead of a PoissonGroup, so Brian2 and
    PyTorch can be fed identical inputs."""

    def __init__(self, sub, stim_global, verbose=True):
        p = SHIU
        self.keep = np.asarray(sub['keep'])
        n = len(self.keep)
        g2l = pd.Series(np.arange(n), index=self.keep)
        self.stim = g2l.loc[np.asarray(stim_global)].to_numpy()
        t0 = time.perf_counter()
        ns = {k: p[k] for k in ('v_0', 'v_rst', 'v_th', 't_mbr', 'tau')}
        self.neu = NeuronGroup(n, EQS, method='linear', threshold='v > v_th',
                               reset='v = v_rst; g = 0*mV', refractory='rfc', namespace=ns)
        self.neu.v = p['v_0']
        self.neu.g = 0 * mV
        self.neu.rfc = p['t_rfc']
        self.neu.rfc[self.stim] = 0 * ms
        self.syn = Synapses(self.neu, self.neu, 'c : 1 (constant)\nw_syn : volt (shared)',
                            on_pre='g_post += c * w_syn', delay=p['t_dly'])
        self.syn.connect(i=sub['pre'], j=sub['post'])
        self.syn.c = sub['w']
        self.syn.w_syn = p['w_syn']
        self.gen = SpikeGeneratorGroup(len(self.stim), [0], [0] * ms)
        self.gen_syn = Synapses(self.gen, self.neu, on_pre='v_post += w_stim',
                                namespace={'w_stim': p['w_syn'] * p['f_poi']})
        self.gen_syn.connect(i=np.arange(len(self.stim)), j=self.stim)
        self.mon = SpikeMonitor(self.neu)
        self.net = Network(self.neu, self.syn, self.gen, self.gen_syn, self.mon)
        self.net.store('init')
        self.build_s = time.perf_counter() - t0
        if verbose:
            print(f"Brian2: {n:,} neurons, {len(sub['w']):,} connections, built in {self.build_s:.1f} s")

    def run(self, input_spikes, w_scale=1.0, verbose=True):
        """input_spikes: list of bool (n_steps, n_stim) arrays, one per trial."""
        dt = defaultclock.dt
        out, self.run_s = [], []
        for trial, inp in enumerate(input_spikes):
            self.net.restore('init')
            step, idx = np.nonzero(inp)
            self.gen.set_spikes(idx, step * dt)
            self.syn.w_syn = SHIU['w_syn'] * w_scale
            t0 = time.perf_counter()
            self.net.run(inp.shape[0] * dt)
            self.run_s.append(time.perf_counter() - t0)
            out.append(pd.DataFrame({'local': np.asarray(self.mon.i),
                                     't': np.asarray(self.mon.t / second), 'trial': trial}))
        self.t_run, self.n_trials = inp.shape[0] * float(dt / second), len(input_spikes)
        if verbose:
            print(f"Brian2: {len(input_spikes)} trials, {np.mean(self.run_s):.1f} s per trial")
        return pd.concat(out, ignore_index=True)

    def rates(self, spikes):
        cnt = np.bincount(spikes['local'], minlength=len(self.keep))
        return pd.Series(cnt / (self.n_trials * self.t_run), index=self.keep, name='rate_hz')


# ---------------------------------------------------------------- comparisons
def _keys(spk, keep, dt_s=1e-4):
    """Spikes as a set-like int64 array of (trial, global neuron, step)."""
    g = np.asarray(keep)[spk['local'].to_numpy()].astype(np.int64)
    step = np.round(spk['t'].to_numpy() / dt_s).astype(np.int64)
    return (spk['trial'].to_numpy().astype(np.int64) * 10**6 + g) * 10**6 + step


def spike_match(a, keep_a, b, keep_b, neurons=None):
    """Compare two spike tables spike by spike (same trial, neuron and time step).
    neurons: optional global indices to restrict the comparison to."""
    ka, kb = _keys(a, keep_a), _keys(b, keep_b)
    if neurons is not None:
        gs = set(np.asarray(neurons).tolist())
        ka = ka[np.isin((ka // 10**6) % 10**6, list(gs))]
        kb = kb[np.isin((kb // 10**6) % 10**6, list(gs))]
    both = np.intersect1d(ka, kb).size
    only_a, only_b = len(ka) - both, len(kb) - both
    diff = np.setxor1d(ka, kb)
    first = (diff % 10**6).min() * 1e-4 if diff.size else None
    return dict(spikes_a=len(ka), spikes_b=len(kb), matched=both,
                only_a=only_a, only_b=only_b,
                match_frac=both / max(len(ka), len(kb)) if (len(ka) or len(kb)) else 1.0,
                first_mismatch_s=first)


def readout(rates, groups):
    """Summary numbers for readout groups. rates: Series indexed by global idx
    (neurons missing from a subnetwork count as 0 Hz)."""
    out = {}
    for name, idx in groups.items():
        r = rates.reindex(idx).fillna(0).to_numpy()
        out[f'{name}_firing'] = int((r >= 1).sum())
        out[f'{name}_mean_hz'] = float(r.mean())
    return out


def rate_agreement(r_ref, r_test, idx):
    """How well per-neuron rates of a test run reproduce a reference run, over idx."""
    a = r_ref.reindex(idx).fillna(0).to_numpy()
    b = r_test.reindex(idx).fillna(0).to_numpy()
    corr = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
    return dict(corr=float(corr), mean_ratio=float(b.mean() / a.mean()) if a.mean() else np.nan,
                mean_abs_diff_hz=float(np.abs(a - b).mean()))


# ---------------------------------------------------------------- subnetworks
def path_subnets(ws, src, dst, Ls, min_syn=5, extra=(), verbose=True):
    """Structural subnetworks: neurons on src -> dst paths of <= L hops (using
    connections with >= min_syn synapses to measure path length), for each L.
    src, dst and extra are always kept. All connections among kept neurons are used."""
    _, df, db = ws.path_neurons(src, dst, max(Ls), min_syn, verbose=False)
    ok = (df >= 0) & (db >= 0)
    total = df.astype(int) + db
    always = np.concatenate([np.asarray(src), np.asarray(dst), np.asarray(extra, dtype=int)])
    subs = {}
    for L in Ls:
        keep = np.union1d(np.flatnonzero(ok & (total <= L)), always)
        subs[f'path L={L}'] = ws.subnetwork(keep)
        if verbose:
            s = subs[f'path L={L}']
            print(f"path L={L}: {len(s['keep']):>7,} neurons, {len(s['w']):>11,} connections")
    return subs


def active_subnet(ws, spike_tables, extra=(), verbose=True):
    """Activity-based subnetwork: every neuron that fired at least once in the
    given full-network runs, plus `extra`. A neuron that never spikes has no
    effect on any other neuron, so for those runs this subnetwork is exact;
    for other stimuli it is only as good as the runs it was built from.
    spike_tables: list of (spikes DataFrame, keep array) pairs."""
    fired = [np.asarray(keep)[np.unique(s['local'])] for s, keep in spike_tables]
    keep = np.union1d(np.concatenate(fired), np.asarray(extra, dtype=int))
    sub = ws.subnetwork(keep)
    if verbose:
        print(f"active set: {len(keep):>7,} neurons, {len(sub['w']):>11,} connections")
    return sub


# ---------------------------------------------------------------- timing
def time_sim_second(fn, reps=3, sim_s=1.0):
    """Median wall-clock seconds per simulated second for fn() (which simulates sim_s)."""
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return float(np.median(t)) / sim_s


# ---------------------------------------------------------------- engines
ENGINES = {
    # name: (description, needs GPU)
    'numba':          ('Numba, CPU, float64, event-driven', False),
    'torch-cpu':      ('PyTorch, CPU, float32, event-driven', False),
    'torch-gpu':      ('PyTorch, GPU, float32, event-driven', True),
    'torch-gpu-f64':  ('PyTorch, GPU, float64, event-driven', True),
    'torch-gpu-spmv': ('PyTorch, GPU, float32, sparse matrix x vector', True),
}


def gpu_available():
    import torch
    return torch.cuda.is_available()


def make_engine(kind, sub, stim_global, verbose=False):
    """Build a simulator by name (see ENGINES)."""
    if kind == 'numba':
        from .lif_numba import NumbaLIF
        return NumbaLIF(sub, stim_global, verbose=verbose)
    import torch
    from .lif_torch import TorchLIF
    dev = 'cuda' if '-gpu' in kind else 'cpu'
    dtype = torch.float64 if kind.endswith('f64') else torch.float32
    mode = 'spmv' if kind.endswith('spmv') else 'event'
    return TorchLIF(sub, stim_global, device=dev, dtype=dtype, mode=mode, verbose=verbose)


def time_engine(eng, rate=150, w_scale=1.0, reps=3, batch=1, sim_s=1.0):
    """Wall-clock seconds per simulated second, per trial (median of reps).
    Includes recording every spike, as a closed loop would need."""
    T = int(round(sim_s * 1000 / eng.dt))
    eng.reset(batch, seed=123, w_scale=w_scale)
    eng.advance(200, rate=rate)                      # warm-up (compile, CUDA init)
    if hasattr(eng, '_sync'): eng._sync()
    t = []
    for r in range(reps):
        eng.reset(batch, seed=r, w_scale=w_scale)
        t0 = time.perf_counter()
        eng.advance(T, rate=rate)
        if hasattr(eng, '_sync'): eng._sync()
        t.append(time.perf_counter() - t0)
    return float(np.median(t)) / sim_s / batch
