"""PyTorch port of sim/lif.py (Shiu et al. 2024 LIF model), for CPU or GPU.

Same model, same parameters, same per-time-step order as Brian2, so with the
same input spikes and float64 it reproduces Brian2's spike trains exactly.

Brian2's default schedule, every dt = 0.1 ms:
  1. groups      integrate v, g exactly over dt ('linear' method), but only
                 for neurons that are not refractory ("unless refractory")
  2. thresholds  spike if v > v_th and not refractory; Poisson input spikes
  3. synapses    deliver recurrent spikes emitted 18 steps ago (1.8 ms delay):
                 g_post += c * w_syn; deliver this step's Poisson spikes:
                 v_post += w_syn * f_poi. Because v and g are declared
                 "(unless refractory)", Brian2 drops any input that arrives
                 while the target is refractory (including the step it spikes).
  4. resets      spiking neurons: v = v_rst, g = 0

Two ways to deliver recurrent spikes:
  'event'  gather only the outgoing connections of neurons that spiked.
           Cost grows with the number of spikes, not with network size.
  'spmv'   one sparse matrix x spike-vector product per step. Cost grows
           with the number of connections; independent of activity.

Several independent trials can run side by side (`batch`): useful for
sweeps and for averaging; a closed loop uses batch = 1.
"""
import math, time

import numpy as np
import pandas as pd
import torch

from .lif import SHIU, peak_mem_gb
from brian2 import mV, ms, Hz, second


def _f(q, unit):
    return float(q / unit)


class TorchLIF:
    def __init__(self, sub, stim_global, device='cpu', dtype=torch.float32, mode='event',
                 params=None, dt_ms=0.1, verbose=True):
        p = dict(SHIU, **(params or {}))
        self.dev = torch.device(device)
        self.dtype = dtype
        self.mode = mode
        self.keep = np.asarray(sub['keep'])
        n = self.n = len(self.keep)
        g2l = pd.Series(np.arange(n), index=self.keep)
        stim = g2l.loc[np.asarray(stim_global)].to_numpy()
        self.stim_local = stim

        # constants (mV, ms)
        self.dt = dt_ms
        self.v_0, self.v_rst, self.v_th = _f(p['v_0'], mV), _f(p['v_rst'], mV), _f(p['v_th'], mV)
        t_mbr, tau = _f(p['t_mbr'], ms), _f(p['tau'], ms)
        self.w_syn = _f(p['w_syn'], mV)
        self.w_stim = _f(SHIU['w_syn'], mV) * SHIU['f_poi']    # fixed, as in lif.py
        self.r_poi = _f(p['r_poi'], Hz)
        # exact solution of dv/dt = (v_0 - v + g)/t_mbr, dg/dt = -g/tau over one step
        self.a = math.exp(-dt_ms / t_mbr)
        self.c = math.exp(-dt_ms / tau)
        self.b = tau / (tau - t_mbr) * (self.c - self.a)
        # step counts, rounded the way Brian2 rounds them
        self.delay = int(round(_f(p['t_dly'], ms) / dt_ms))
        rfc = int((_f(p['t_rfc'], ms) + 1e-3 * dt_ms) / dt_ms)
        rfc_steps = np.full(n, rfc, np.int32)
        rfc_steps[stim] = 0                                     # as in lif.py / Shiu

        t0 = time.perf_counter()
        kw = dict(device=self.dev)
        self.rfc_steps = torch.tensor(rfc_steps, **kw)
        self.stim = torch.tensor(stim, dtype=torch.long, **kw)
        pre, post = np.asarray(sub['pre']), np.asarray(sub['post'])
        w = np.asarray(sub['w'])
        self.n_conn = len(w)
        if mode == 'event':
            # CSR by presynaptic neuron, connections kept in their original order
            order = np.argsort(pre, kind='stable')
            indptr = np.zeros(n + 1, np.int64)
            np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
            self.indptr = torch.tensor(indptr, **kw)
            self.post = torch.tensor(post[order].astype(np.int64), **kw)
            self.c_w = torch.tensor(w[order], dtype=dtype, **kw)
        elif mode == 'spmv':
            # W^T as CSR (rows = post), so g_in = W^T @ spikes
            order = np.lexsort((pre, post))
            crow = np.zeros(n + 1, np.int64)
            np.cumsum(np.bincount(post, minlength=n), out=crow[1:])
            self.WT = torch.sparse_csr_tensor(
                torch.tensor(crow, **kw), torch.tensor(pre[order].astype(np.int64), **kw),
                torch.tensor(w[order], dtype=dtype, **kw), size=(n, n))
        else:
            raise ValueError(mode)
        self._sync()
        self.build_s = time.perf_counter() - t0
        if verbose:
            print(f"torch {mode} on {self.dev} ({str(dtype).split('.')[-1]}): {n:,} neurons, "
                  f"{self.n_conn:,} connections, built in {self.build_s:.1f} s")
        self.reset(1)

    # ------------------------------------------------------------ state
    def _sync(self):
        if self.dev.type == 'cuda':
            torch.cuda.synchronize(self.dev)

    def reset(self, batch=1, seed=0, w_scale=1.0):
        """Fresh state for `batch` independent trials."""
        n, kw = self.n, dict(device=self.dev)
        self.B = batch
        self.v = torch.full((batch, n), self.v_0, dtype=self.dtype, **kw)
        self.g = torch.zeros((batch, n), dtype=self.dtype, **kw)
        self.last = torch.full((batch, n), -10**9, dtype=torch.int32, **kw)
        self.step_i = 0
        self.w_eff = self.w_syn * w_scale
        self.ring = [None] * self.delay        # spikes waiting to be delivered
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)

    # ------------------------------------------------------------ one step
    def _deliver(self, flat, open_):
        """g_post += c * w_eff for every connection of neurons in `flat` (b*n + i),
        skipping targets that are refractory (open_ = not refractory, as a float mask)."""
        n = self.n
        if self.mode == 'event':
            i = flat % n
            starts = self.indptr[i]
            cnt = self.indptr[i + 1] - starts
            total = int(cnt.sum())
            if total == 0:
                return
            rep = torch.repeat_interleave(torch.arange(len(i), device=self.dev), cnt,
                                          output_size=total)
            first = torch.cumsum(cnt, 0) - cnt
            e = starts[rep] + torch.arange(total, device=self.dev) - first[rep]
            tgt = self.post[e] + (flat // n)[rep] * n
            self.g.view(-1).index_add_(0, tgt, self.c_w[e] * open_.view(-1)[tgt] * self.w_eff)
        else:
            s = torch.zeros(self.B * n, dtype=self.dtype, device=self.dev)
            s[flat] = 1
            gin = torch.sparse.mm(self.WT, s.view(self.B, n).T)      # (n, B)
            self.g += gin.T * open_ * self.w_eff

    def advance(self, n_steps, rate=None, input_spikes=None, record=True):
        """Advance all trials by n_steps.

        rate: stimulus rate in Hz (scalar, or one value per stimulated neuron,
              or a (batch, n_stim) array). Default: r_poi.
        input_spikes: optional bool (n_steps, batch, n_stim) array of stimulus
              spikes to use instead of Poisson draws (for exact comparisons).
        Returns the recorded spikes as flat indices (b*n + i) and step numbers.
        """
        B, n, dev = self.B, self.n, self.dev
        if input_spikes is None:
            r = self.r_poi if rate is None else rate
            prob = torch.as_tensor(np.asarray(r, dtype=np.float64) * self.dt * 1e-3,
                                   dtype=self.dtype, device=dev)
            prob = prob.expand(B, len(self.stim)).contiguous()
        else:
            input_spikes = torch.as_tensor(input_spikes, device=dev)
        rec_i, rec_t = [], []
        for k in range(n_steps):
            t = self.step_i
            # 1. integrate (frozen while refractory)
            nr = (t - self.last) >= self.rfc_steps
            v_new = self.v_0 + (self.v - self.v_0) * self.a + self.g * self.b
            self.v = torch.where(nr, v_new, self.v)
            self.g = torch.where(nr, self.g * self.c, self.g)
            # 2. threshold
            spk = (self.v > self.v_th) & nr
            flat = spk.view(-1).nonzero().squeeze(1)
            self.last.view(-1)[flat] = t
            if input_spikes is None:
                stim_spk = torch.rand((B, len(self.stim)), generator=self.gen, device=dev,
                                      dtype=self.dtype) < prob
            else:
                stim_spk = input_spikes[k]
            # 3. synapses: recurrent spikes from `delay` steps ago, then stimulus;
            #    refractory targets (incl. neurons spiking now) ignore input
            open_ = (nr & ~spk).to(self.dtype)
            old = self.ring[t % self.delay]
            if old is not None and len(old):
                self._deliver(old, open_)
            self.ring[t % self.delay] = flat
            self.v[:, self.stim] += stim_spk.to(self.dtype) * open_[:, self.stim] * self.w_stim
            # 4. reset
            self.v.view(-1)[flat] = self.v_rst
            self.g.view(-1)[flat] = 0
            if record:
                rec_i.append(flat)
                rec_t.append(torch.full_like(flat, t))
            self.step_i += 1
        if not record:
            return None
        return torch.cat(rec_i), torch.cat(rec_t)

    # ------------------------------------------------------------ trials (same API as LIFNet)
    def run(self, rate=None, n_trials=5, t_run=1 * second, w_scale=1.0, seed=0, batch=None,
            input_spikes=None, verbose=True):
        """Run trials; returns spikes (local idx, time, trial) as a DataFrame.
        batch: trials run side by side (default: all at once).
        input_spikes: optional list (one per trial) of bool (n_steps, n_stim) arrays."""
        if input_spikes is not None:
            n_trials = len(input_spikes)
            t_run = input_spikes[0].shape[0] * self.dt * 1e-3 * second
        T = int(round(float(t_run / second) * 1000 / self.dt))
        batch = n_trials if batch is None else batch
        out = []
        t0 = time.perf_counter()
        for first in range(0, n_trials, batch):
            nb = min(batch, n_trials - first)
            self.reset(nb, seed=seed + first, w_scale=w_scale)
            inp = None
            if input_spikes is not None:
                inp = np.stack([input_spikes[first + j] for j in range(nb)], axis=1)
            fi, ft = self.advance(T, rate=rate, input_spikes=inp)
            fi, ft = fi.cpu().numpy(), ft.cpu().numpy()
            out.append(pd.DataFrame({'local': fi % self.n, 't': ft * self.dt * 1e-3,
                                     'trial': fi // self.n + first}))
        self._sync()
        self.wall_s = time.perf_counter() - t0
        self.t_run, self.n_trials = float(t_run / second), n_trials
        if verbose:
            print(f"{n_trials} trials × {self.t_run:g} s: {self.wall_s:.1f} s "
                  f"({self.wall_s / (n_trials * self.t_run):.2f} s per simulated second per trial)")
        return pd.concat(out, ignore_index=True)

    def rates(self, spikes, ws=None):
        """Mean firing rate (Hz) per neuron over trials, indexed by global idx."""
        cnt = np.bincount(spikes['local'], minlength=self.n)
        r = pd.Series(cnt / (self.n_trials * self.t_run), index=self.keep, name='rate_hz')
        if ws is not None:
            r = ws.neurons.loc[self.keep, ['bodyId', 'type', 'cls', 'nt_used']].assign(rate_hz=r.values)
        return r
