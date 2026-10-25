"""LIF connectome model on the male-CNS v1.0 weight set.

Neuron and synapse model copied from Shiu et al. 2024, model.py
(github.com/philshiu/Drosophila_brain_model). Changes from their code:
  - weights are stored as signed synapse counts; w_syn is a shared synapse
    variable, so a w_syn sweep needs no rebuild
  - the network is built once and reset between trials (store/restore),
    instead of being rebuilt every trial
  - stimulus is a PoissonGroup with one-to-one synapses (same effect as their
    per-neuron PoissonInput, N=1), and its strength is fixed at Shiu's
    w_syn * f_poi so it does not change during the w_syn sweep
  - networks can be built on a subnetwork (to fit in 4 GB)
"""
from pathlib import Path
import json, time, resource

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from brian2 import (NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network,
                    mV, ms, Hz, second, seed as b2seed)

# ---------------------------------------------------------------- parameters
# Shiu et al. model.py, default_params
SHIU = dict(
    v_0=-52 * mV,      # resting potential
    v_rst=-52 * mV,    # reset potential after spike
    v_th=-45 * mV,     # spike threshold
    t_mbr=20 * ms,     # membrane time constant
    tau=5 * ms,        # synaptic time constant
    t_rfc=2.2 * ms,    # refractory period
    t_dly=1.8 * ms,    # synaptic delay
    w_syn=0.275 * mV,  # weight per synapse (free parameter, tuned on FlyWire)
    r_poi=150 * Hz,    # stimulus rate
    f_poi=250,         # stimulus synapse = w_syn * f_poi (always suprathreshold)
)

EQS = '''
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau : volt (unless refractory)
rfc : second
'''


def peak_mem_gb():
    """Peak memory of this Python process so far (Linux reports kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ---------------------------------------------------------------- weight set
class WeightSet:
    """Loads weights_male-cns_v1.0/ (made in 02_weights.ipynb)."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / 'meta.json').read_text())
        self.neurons = pd.read_parquet(self.folder / 'neurons_signs.parquet')
        e = pq.read_table(self.folder / 'edges.parquet')
        self.pre = e['pre_idx'].to_numpy()
        self.post = e['post_idx'].to_numpy()
        self.count = e['syn_count'].to_numpy()
        self.N = len(self.neurons)
        self._out = None   # rows = pre, cols = post, synapse counts

    def signed(self, sign_col='sign'):
        """Signed synapse count for every edge ('sign' or 'sign_modexc')."""
        return self.count * self.neurons[sign_col].to_numpy()[self.pre].astype(np.int32)

    def idx(self, mask):
        """Neuron indices where a boolean mask over ws.neurons is True."""
        return self.neurons.index[np.asarray(mask)].to_numpy()

    def by_type(self, pattern, cls=None):
        """Neuron indices whose type matches a regex (optionally within a class)."""
        m = self.neurons['type'].astype(str).str.match(pattern)
        if cls is not None:
            m &= self.neurons['cls'] == cls
        return self.idx(m)

    def _adj(self):
        if self._out is None:
            self._out = sp.csr_matrix((self.count, (self.pre, self.post)),
                                      shape=(self.N, self.N))
        return self._out

    def partners(self, idx, direction='out', min_syn=1):
        """Direct partners of a set of neurons, with summed synapse counts."""
        A = self._adj()
        A = A if direction == 'out' else A.T.tocsr()
        sub = A[np.asarray(idx)]
        tot = np.asarray(sub.sum(axis=0)).ravel()
        keep = np.flatnonzero(tot >= min_syn)
        return pd.Series(tot[keep], index=keep, name='syn_from_group')

    def distances(self, seeds, n_hops, direction='out', min_syn=1, verbose=True):
        """Hop distance from seeds for every neuron (-1 = not reached within n_hops).
        Each hop uses only connections with >= min_syn synapses."""
        A = self._adj()
        A = A if direction == 'out' else A.T.tocsr()
        A = A.multiply(A >= min_syn).tocsr()
        dist = np.full(self.N, -1, np.int16)
        dist[seeds] = 0
        frontier = np.asarray(seeds)
        for h in range(1, n_hops + 1):
            nxt = np.unique(A[frontier].indices)
            frontier = nxt[dist[nxt] < 0]
            dist[frontier] = h
            if verbose:
                print(f"  hop {h}: +{len(frontier):,} new, {(dist >= 0).sum():,} total")
        return dist

    def hops(self, seeds, n_hops, direction='out', min_syn=1, verbose=True):
        """Neurons reachable within n_hops (each hop: edges with >= min_syn)."""
        return np.flatnonzero(self.distances(seeds, n_hops, direction, min_syn, verbose) >= 0)

    def path_neurons(self, src, dst, max_len, min_syn=1, verbose=True):
        """Neurons on some path src -> ... -> dst of at most max_len hops:
        dist_from_src + dist_to_dst <= max_len."""
        if verbose: print("downstream of source:")
        df = self.distances(src, max_len, 'out', min_syn, verbose)
        if verbose: print("upstream of target:")
        db = self.distances(dst, max_len, 'in', min_syn, verbose)
        ok = (df >= 0) & (db >= 0) & (df.astype(int) + db <= max_len)
        return np.flatnonzero(ok), df, db

    def subnetwork(self, keep, sign_col='sign'):
        """Edges among `keep` only, renumbered 0..len(keep)-1."""
        keep = np.unique(keep)
        local = np.full(self.N, -1, np.int64)
        local[keep] = np.arange(len(keep))
        m = (local[self.pre] >= 0) & (local[self.post] >= 0)
        w = self.signed(sign_col)[m]
        nz = w != 0
        return dict(keep=keep,
                    pre=local[self.pre[m]][nz].astype(np.int32),
                    post=local[self.post[m]][nz].astype(np.int32),
                    w=w[nz].astype(np.float64))


# ---------------------------------------------------------------- simulation
class LIFNet:
    """Brian2 LIF network on a subnetwork (or the full set: keep=all indices)."""

    def __init__(self, sub, stim_global, params=None, verbose=True):
        self.p = dict(SHIU, **(params or {}))
        self.keep = sub['keep']
        n = len(self.keep)
        g2l = pd.Series(np.arange(n), index=self.keep)
        self.stim = g2l.loc[np.asarray(stim_global)].to_numpy()

        t0 = time.perf_counter()
        ns = {k: self.p[k] for k in ('v_0', 'v_rst', 'v_th', 't_mbr', 'tau')}
        self.neu = NeuronGroup(n, EQS, method='linear', threshold='v > v_th',
                               reset='v = v_rst; g = 0*mV', refractory='rfc',
                               namespace=ns, name='neurons')
        self.neu.v = self.p['v_0']
        self.neu.g = 0 * mV
        self.neu.rfc = self.p['t_rfc']
        self.neu.rfc[self.stim] = 0 * ms          # as in Shiu: stimulated neurons not refractory

        self.syn = Synapses(self.neu, self.neu,
                            'c : 1 (constant)\nw_syn : volt (shared)',
                            on_pre='g_post += c * w_syn',
                            delay=self.p['t_dly'], name='synapses')
        self.syn.connect(i=sub['pre'], j=sub['post'])
        self.syn.c = sub['w']
        self.syn.w_syn = self.p['w_syn']

        self.poi = PoissonGroup(len(self.stim), rates=0 * Hz, name='stimulus')
        self.poi_syn = Synapses(self.poi, self.neu, on_pre='v_post += w_stim',
                                namespace={'w_stim': SHIU['w_syn'] * SHIU['f_poi']},
                                name='stim_synapses')
        self.poi_syn.connect(i=np.arange(len(self.stim)), j=self.stim)

        self.mon = SpikeMonitor(self.neu, name='spikes')
        self.net = Network(self.neu, self.syn, self.poi, self.poi_syn, self.mon)
        self.net.store('init')
        if verbose:
            print(f"built: {n:,} neurons, {len(sub['w']):,} connections, "
                  f"{time.perf_counter() - t0:.1f} s, peak memory {peak_mem_gb():.2f} GB")

    def run(self, rate=None, n_trials=5, t_run=1 * second, w_scale=1.0, seed=0, verbose=True):
        """Run trials; returns spikes (local idx, time, trial) as a DataFrame."""
        rate = self.p['r_poi'] if rate is None else rate
        out = []
        t0 = time.perf_counter()
        for trial in range(n_trials):
            self.net.restore('init')
            b2seed(seed + trial)
            self.syn.w_syn = self.p['w_syn'] * w_scale
            self.poi.rates = rate
            self.net.run(t_run)
            out.append(pd.DataFrame({'local': np.asarray(self.mon.i),
                                     't': np.asarray(self.mon.t / second),
                                     'trial': trial}))
        self.t_run, self.n_trials = float(t_run / second), n_trials
        if verbose:
            print(f"{n_trials} trials × {self.t_run:g} s: {time.perf_counter() - t0:.1f} s, "
                  f"peak memory {peak_mem_gb():.2f} GB")
        return pd.concat(out, ignore_index=True)

    def rates(self, spikes, ws=None):
        """Mean firing rate (Hz) per neuron over trials, indexed by global idx."""
        cnt = np.bincount(spikes['local'], minlength=len(self.keep))
        r = pd.Series(cnt / (self.n_trials * self.t_run), index=self.keep, name='rate_hz')
        if ws is not None:
            r = ws.neurons.loc[self.keep, ['bodyId', 'type', 'cls', 'nt_used']].assign(rate_hz=r.values)
        return r
