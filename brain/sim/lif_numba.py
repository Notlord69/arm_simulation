"""Numba CPU engine for the same LIF model (sim/lif.py), for machines without a GPU.

Same model, parameters and per-step order as Brian2 (see lif_torch.py), but
each time step is one compiled loop over the neurons instead of ~30 separate
array operations, so it is much faster on a CPU. float64 throughout.

Same API as TorchLIF: reset(batch, seed, w_scale), advance(n_steps, rate,
input_spikes), run(...), rates(...).
"""
import math, time

import numpy as np
import pandas as pd
from numba import njit
from brian2 import mV, ms, Hz, second

from .lif import SHIU


@njit(cache=True)
def _kernel(n_steps, t0, v, g, last, open_, rfc, indptr, post, cw, w_eff,
            ring, ring_n, delay, stim, prob, inp, use_inp, w_stim,
            v_0, v_rst, v_th, a, b, c, rec_i, rec_t, n_rec, seed):
    """Advance one trial. Returns (steps done, spikes recorded). Stops early if
    the record buffer is nearly full; the caller grows it and calls again."""
    if seed >= 0:
        np.random.seed(seed)
    n = v.shape[0]
    cap = rec_i.shape[0]
    spk_now = np.empty(n, np.int32)
    for k in range(n_steps):
        if n_rec + n > cap:
            return k, n_rec
        t = t0 + k
        # 1. integrate unless refractory; 2. threshold
        ns = 0
        for i in range(n):
            nr = (t - last[i]) >= rfc[i]
            if nr:
                v[i] = v_0 + (v[i] - v_0) * a + g[i] * b
                g[i] = g[i] * c
                if v[i] > v_th:
                    last[i] = t
                    spk_now[ns] = i
                    ns += 1
                    open_[i] = False
                else:
                    open_[i] = True
            else:
                open_[i] = False
        # 3a. recurrent spikes emitted `delay` steps ago
        slot = t % delay
        for s in range(ring_n[slot]):
            pre = ring[slot, s]
            for e in range(indptr[pre], indptr[pre + 1]):
                j = post[e]
                if open_[j]:
                    g[j] += cw[e] * w_eff
        for s in range(ns):
            ring[slot, s] = spk_now[s]
        ring_n[slot] = ns
        # 3b. stimulus
        for q in range(stim.shape[0]):
            if use_inp:
                fire = inp[k, q]
            else:
                fire = np.random.random() < prob[q]
            j = stim[q]
            if fire and open_[j]:
                v[j] += w_stim
        # 4. reset, record
        for s in range(ns):
            i = spk_now[s]
            v[i] = v_rst
            g[i] = 0.0
            rec_i[n_rec] = i
            rec_t[n_rec] = t
            n_rec += 1
    return n_steps, n_rec


def _f(q, unit):
    return float(q / unit)


class NumbaLIF:
    def __init__(self, sub, stim_global, params=None, dt_ms=0.1, verbose=True):
        p = dict(SHIU, **(params or {}))
        self.mode, self.dev = 'numba', 'cpu'
        self.keep = np.asarray(sub['keep'])
        n = self.n = len(self.keep)
        g2l = pd.Series(np.arange(n), index=self.keep)
        self.stim = g2l.loc[np.asarray(stim_global)].to_numpy().astype(np.int64)
        self.dt = dt_ms
        self.v_0, self.v_rst, self.v_th = _f(p['v_0'], mV), _f(p['v_rst'], mV), _f(p['v_th'], mV)
        t_mbr, tau = _f(p['t_mbr'], ms), _f(p['tau'], ms)
        self.w_syn = _f(p['w_syn'], mV)
        self.w_stim = _f(SHIU['w_syn'], mV) * SHIU['f_poi']
        self.r_poi = _f(p['r_poi'], Hz)
        self.a = math.exp(-dt_ms / t_mbr)
        self.c = math.exp(-dt_ms / tau)
        self.b = tau / (tau - t_mbr) * (self.c - self.a)
        self.delay = int(round(_f(p['t_dly'], ms) / dt_ms))
        rfc = int((_f(p['t_rfc'], ms) + 1e-3 * dt_ms) / dt_ms)
        self.rfc = np.full(n, rfc, np.int64)
        self.rfc[self.stim] = 0

        t0 = time.perf_counter()
        pre, post, w = np.asarray(sub['pre']), np.asarray(sub['post']), np.asarray(sub['w'])
        self.n_conn = len(w)
        order = np.argsort(pre, kind='stable')
        self.indptr = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(pre, minlength=n), out=self.indptr[1:])
        self.post = post[order].astype(np.int32)
        self.cw = w[order].astype(np.float64)
        self.build_s = time.perf_counter() - t0
        self.reset(1)
        self.advance(1, record=False)           # compile (cached on disk after the first time)
        self.reset(1)
        if verbose:
            print(f"numba on cpu: {n:,} neurons, {self.n_conn:,} connections, "
                  f"built in {self.build_s:.1f} s")

    def reset(self, batch=1, seed=0, w_scale=1.0):
        n = self.n
        self.B = batch
        self.v = np.full((batch, n), self.v_0)
        self.g = np.zeros((batch, n))
        self.last = np.full((batch, n), -10**9, np.int64)
        self.open_ = np.ones((batch, n), np.bool_)
        self.ring = np.zeros((batch, self.delay, n), np.int32)
        self.ring_n = np.zeros((batch, self.delay), np.int64)
        self.step_i = 0
        self.w_eff = self.w_syn * w_scale
        self.seeds = [seed + b for b in range(batch)]   # seeds numba's RNG on the next advance

    def advance(self, n_steps, rate=None, input_spikes=None, record=True):
        """Same as TorchLIF.advance. Returns (flat idx b*n + i, step) numpy arrays."""
        r = self.r_poi if rate is None else rate
        prob = np.broadcast_to(np.asarray(r, np.float64) * self.dt * 1e-3,
                               (self.B, len(self.stim))).copy()
        out_i, out_t = [], []
        for bt in range(self.B):
            if input_spikes is not None:
                inp, use = np.ascontiguousarray(np.asarray(input_spikes)[:, bt, :], np.bool_), True
            else:
                inp, use = np.zeros((1, 1), np.bool_), False
            cap = max(4 * self.n, 1 << 20)
            rec_i, rec_t = np.empty(cap, np.int64), np.empty(cap, np.int64)
            done, n_rec, seed = 0, 0, self.seeds[bt]
            while done < n_steps:
                k, n_rec = _kernel(n_steps - done, self.step_i + done, self.v[bt], self.g[bt],
                                   self.last[bt], self.open_[bt], self.rfc, self.indptr, self.post,
                                   self.cw, self.w_eff, self.ring[bt], self.ring_n[bt], self.delay,
                                   self.stim, prob[bt], inp[done:] if use else inp, use,
                                   self.w_stim, self.v_0, self.v_rst, self.v_th, self.a, self.b,
                                   self.c, rec_i, rec_t, n_rec, seed)
                seed = -1                           # RNG continues from here on
                done += k
                if done < n_steps:                  # buffer nearly full: grow it
                    rec_i = np.concatenate([rec_i, np.empty_like(rec_i)])
                    rec_t = np.concatenate([rec_t, np.empty_like(rec_t)])
            self.seeds[bt] = -1
            out_i.append(rec_i[:n_rec] + bt * self.n)
            out_t.append(rec_t[:n_rec])
        self.step_i += n_steps
        if not record:
            return None
        return np.concatenate(out_i), np.concatenate(out_t)

    def run(self, rate=None, n_trials=5, t_run=1 * second, w_scale=1.0, seed=0, batch=None,
            input_spikes=None, verbose=True):
        if input_spikes is not None:
            n_trials = len(input_spikes)
            t_run = input_spikes[0].shape[0] * self.dt * 1e-3 * second
        T = int(round(float(t_run / second) * 1000 / self.dt))
        out = []
        t0 = time.perf_counter()
        for trial in range(n_trials):
            self.reset(1, seed=seed + trial, w_scale=w_scale)
            inp = None if input_spikes is None else input_spikes[trial][:, None, :]
            fi, ft = self.advance(T, rate=rate, input_spikes=inp)
            out.append(pd.DataFrame({'local': fi, 't': ft * self.dt * 1e-3, 'trial': trial}))
        self.wall_s = time.perf_counter() - t0
        self.t_run, self.n_trials = float(t_run / second), n_trials
        if verbose:
            print(f"{n_trials} trials × {self.t_run:g} s: {self.wall_s:.1f} s "
                  f"({self.wall_s / (n_trials * self.t_run):.2f} s per simulated second per trial)")
        return pd.concat(out, ignore_index=True)

    def rates(self, spikes, ws=None):
        cnt = np.bincount(spikes['local'], minlength=self.n)
        r = pd.Series(cnt / (self.n_trials * self.t_run), index=self.keep, name='rate_hz')
        if ws is not None:
            r = ws.neurons.loc[self.keep, ['bodyId', 'type', 'cls', 'nt_used']].assign(rate_hz=r.values)
        return r
