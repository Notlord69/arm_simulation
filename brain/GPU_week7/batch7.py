"""Week 7: batched brain engines - many closed-loop episodes side by side.

Why batch: on a GPU, one episode at a time is SLOWER than the laptop (week 3: 10.9 s of compute
per simulated second, almost all fixed overhead per 0.1 ms step), but 32 episodes side by side
cost about 1.2 s each. So every stage runs its episodes as one batch: a DAgger round (30), the
test (50), the timing sweep (10), the encoder sweep (16 trials) and the activity matching (12 at
a time).

Engines (same per-step order and maths as lif_numba.py / lif_torch.py / Brian2):
  TorchBatchLIF   torch on CUDA (or CPU, for testing); per-trial w_scale; trials that finish are
                  dropped from the batch, so the GPU only works on episodes still running
  NumbaBatchLIF   lif_numba's compiled kernel, one trial at a time (the laptop path)

The Poisson input is drawn from a counter-based hash of (episode seed, time step, input neuron).
So an episode's input spikes don't depend on which batch it ran in or on the engine. With
float64, torch-CPU and numba produce identical spikes (checked in the sandbox). On CUDA,
synaptic inputs are summed with atomic adds in a varying order, so float64 can differ by rare
last-digit rounding, and float32 diverges sooner; rates agree (week 3), and the GPU check cell in
the notebook measures this on the real network.
"""
import math, time

import numpy as np

M32 = 0xFFFFFFFF
_K = 0x45D9F3B                    # < 2^27, so products of 32-bit values stay inside int64
_GOLD = 0x9E3779B1


def _h(x):
    """32-bit integer hash (works on numpy int64 arrays/scalars and torch int64 tensors)."""
    x = (((x >> 16) ^ x) * _K) & M32
    x = (((x >> 16) ^ x) * _K) & M32
    return (x >> 16) ^ x


def _seed_base(seeds, n_stim):
    """(B, n_stim) int64: one hash per (episode seed, input neuron)."""
    sb = _h(np.asarray(seeds, np.int64) & M32)
    j = (((np.arange(n_stim, dtype=np.int64) + 1) * _GOLD) & M32)
    return _h(sb[:, None] ^ j[None, :])


def _step_hash(t):
    """Hash of the global step numbers t (int64 array)."""
    return _h((np.asarray(t, np.int64) * 0x85EBCA6B + 0x27D4EB2F) & M32)


def poisson_input_np(base, t0, n_steps, prob):
    """bool (n_steps, B, n_stim): stimulus spikes for steps t0..t0+n_steps-1.
    base (B, n_stim) from _seed_base, prob (B, n_stim) = rate * dt."""
    ht = _step_hash(np.arange(t0, t0 + n_steps))
    u = _h(base[None, :, :] ^ ht[:, None, None])
    return u < np.round(np.asarray(prob) * 2.0 ** 32).astype(np.int64)[None]


def _consts(p, dt_ms):
    from brian2 import mV, ms, Hz
    f = lambda q, u: float(q / u)
    c = dict(v_0=f(p['v_0'], mV), v_rst=f(p['v_rst'], mV), v_th=f(p['v_th'], mV),
             w_syn=f(p['w_syn'], mV), r_poi=f(p['r_poi'], Hz))
    t_mbr, tau = f(p['t_mbr'], ms), f(p['tau'], ms)
    c['a'] = math.exp(-dt_ms / t_mbr)
    c['c'] = math.exp(-dt_ms / tau)
    c['b'] = tau / (tau - t_mbr) * (c['c'] - c['a'])
    c['delay'] = int(round(f(p['t_dly'], ms) / dt_ms))
    c['rfc'] = int((f(p['t_rfc'], ms) + 1e-3 * dt_ms) / dt_ms)
    return c


# ---------------------------------------------------------------- torch (GPU)
class TorchBatchLIF:
    """Batched LIF engine in torch. State is (B, n); B changes as trials are dropped (keep())."""
    kind = 'torch'

    def __init__(self, sub, stim_global, device='cuda', dtype='float32', params=None, dt_ms=0.1,
                 verbose=True):
        import torch
        from .lif import SHIU
        self.torch = torch
        p = dict(SHIU, **(params or {}))
        self.dev = torch.device(device)
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.keep_ids = np.asarray(sub['keep'])
        n = self.n = len(self.keep_ids)
        self.stim_np = np.searchsorted(self.keep_ids, np.asarray(stim_global))
        assert np.array_equal(self.keep_ids[self.stim_np], np.asarray(stim_global)), "inputs not in network"
        self.dt = dt_ms
        c = _consts(p, dt_ms)
        for k in ('v_0', 'v_rst', 'v_th', 'w_syn', 'r_poi', 'a', 'b', 'c', 'delay'):
            setattr(self, k, c[k])
        from brian2 import mV
        self.w_stim = float(SHIU['w_syn'] / mV) * SHIU['f_poi']
        rfc = np.full(n, c['rfc'], np.int32); rfc[self.stim_np] = 0
        t0 = time.perf_counter()
        kw = dict(device=self.dev)
        self.rfc_steps = torch.tensor(rfc, **kw)
        self.stim = torch.tensor(self.stim_np, dtype=torch.long, **kw)
        pre, post, w = np.asarray(sub['pre']), np.asarray(sub['post']), np.asarray(sub['w'])
        order = np.argsort(pre, kind='stable')
        indptr = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
        self.indptr = torch.tensor(indptr, **kw)
        self.post = torch.tensor(post[order].astype(np.int64), **kw)
        self.c_w = torch.tensor(w[order], dtype=self.dtype, **kw)
        self.n_conn = len(w)
        del order
        self._sync()
        self.build_s = time.perf_counter() - t0
        if verbose:
            print(f"torch on {self.dev} ({str(self.dtype).split('.')[-1]}): {n:,} neurons, "
                  f"{self.n_conn:,} connections, built in {self.build_s:.1f} s")

    def _sync(self):
        if self.dev.type == 'cuda':
            self.torch.cuda.synchronize(self.dev)

    def reset(self, seeds, w_scales, track=False):
        torch, n = self.torch, self.n
        B = self.B = len(seeds)
        kw = dict(device=self.dev)
        self.v = torch.full((B, n), self.v_0, dtype=self.dtype, **kw)
        self.g = torch.zeros((B, n), dtype=self.dtype, **kw)
        self.last = torch.full((B, n), -10 ** 9, dtype=torch.int32, **kw)
        self.ring = [torch.zeros(0, dtype=torch.long, **kw) for _ in range(self.delay)]
        self.step_i = 0
        self.w_eff = torch.tensor(np.asarray(w_scales, float) * self.w_syn, dtype=self.dtype, **kw)
        self.base = torch.tensor(_seed_base(seeds, len(self.stim_np)), **kw)
        self.track = track
        self.tot = torch.zeros((B, n), dtype=torch.int32, **kw) if track else None

    def keep(self, idx):
        """Keep only trials idx (positions in the current batch, ascending)."""
        torch, n = self.torch, self.n
        idx_t = torch.as_tensor(np.asarray(idx, np.int64), device=self.dev)
        remap = torch.full((self.B,), -1, dtype=torch.long, device=self.dev)
        remap[idx_t] = torch.arange(len(idx), device=self.dev)
        for k in ('v', 'g', 'last', 'w_eff', 'base') + (('tot',) if self.track else ()):
            setattr(self, k, getattr(self, k)[idx_t])
        for s, fl in enumerate(self.ring):
            if len(fl):
                nb = remap[fl // n]
                m = nb >= 0
                self.ring[s] = nb[m] * n + fl[m] % n
        self.B = len(idx)

    def advance(self, prob, n_steps, rec_local, bin_steps):
        """prob: (B, n_stim) spike probability per step. Returns spike counts of the neurons
        rec_local (local indices) in bins of bin_steps steps: numpy uint16 (B, n_bins, n_rec)."""
        torch, n, B = self.torch, self.n, self.B
        dev = self.dev
        rec = torch.as_tensor(np.asarray(rec_local, np.int64), device=dev)
        thr = torch.as_tensor(np.round(np.asarray(prob) * 2.0 ** 32).astype(np.int64), device=dev)[None]
        buf = torch.zeros((n_steps, B, len(rec)), dtype=torch.bool, device=dev)
        CH = 400                     # stimulus spikes drawn 40 ms at a time (few kernels, bounded memory)
        for k in range(n_steps):
            if k % CH == 0:
                ht = torch.as_tensor(_step_hash(np.arange(self.step_i, self.step_i + min(CH, n_steps - k))),
                                     device=dev)
                inp = _h(self.base[None] ^ ht[:, None, None]) < thr                # (<=CH, B, n_stim)
            t = self.step_i
            nr = (t - self.last) >= self.rfc_steps
            v_new = self.v_0 + (self.v - self.v_0) * self.a + self.g * self.b
            self.v = torch.where(nr, v_new, self.v)
            self.g = torch.where(nr, self.g * self.c, self.g)
            spk = (self.v > self.v_th) & nr
            flat = spk.view(-1).nonzero().squeeze(1)
            self.last.view(-1)[flat] = t
            open_ = (nr & ~spk).to(self.dtype)
            old = self.ring[t % self.delay]
            if len(old):
                i = old % n
                starts = self.indptr[i]
                cnt = self.indptr[i + 1] - starts
                total = int(cnt.sum())
                if total:
                    rep = torch.repeat_interleave(torch.arange(len(i), device=dev), cnt, output_size=total)
                    first = torch.cumsum(cnt, 0) - cnt
                    e = starts[rep] + torch.arange(total, device=dev) - first[rep]
                    bb = (old // n)[rep]
                    tgt = self.post[e] + bb * n
                    self.g.view(-1).index_add_(0, tgt, self.c_w[e] * open_.view(-1)[tgt] * self.w_eff[bb])
            self.ring[t % self.delay] = flat
            self.v[:, self.stim] += inp[k % CH].to(self.dtype) * open_[:, self.stim] * self.w_stim
            self.v.view(-1)[flat] = self.v_rst
            self.g.view(-1)[flat] = 0
            buf[k] = spk[:, rec]
            if self.track:
                self.tot += spk
            self.step_i += 1
        nb = n_steps // bin_steps
        counts = buf.view(nb, bin_steps, B, len(rec)).sum(1, dtype=torch.int32).permute(1, 0, 2)
        return counts.to(torch.int32).cpu().numpy().astype(np.uint16)

    def totals(self):
        return self.tot.cpu().numpy()


# ---------------------------------------------------------------- numba (CPU, laptop)
class NumbaBatchLIF:
    """lif_numba's kernel, looping over trials. Same API as TorchBatchLIF. Use batch 1 on the
    laptop (memory): there is no speed gain from batching on a CPU."""
    kind = 'numba'

    def __init__(self, sub, stim_global, params=None, dt_ms=0.1, verbose=True):
        from .lif_numba import NumbaLIF
        self.eng = NumbaLIF(sub, stim_global, params=params, dt_ms=dt_ms, verbose=verbose)
        self.n, self.dt = self.eng.n, dt_ms
        self.keep_ids = self.eng.keep
        self.stim_np = self.eng.stim

    def reset(self, seeds, w_scales, track=False):
        e = self.eng
        e.reset(len(seeds), seed=0, w_scale=1.0)
        self.B = len(seeds)
        self.w_eff = np.asarray(w_scales, float) * e.w_syn
        self.base = _seed_base(seeds, len(self.stim_np))
        self.step_i = 0
        self.track = track
        self.tot = np.zeros((self.B, self.n), np.int64) if track else None

    def keep(self, idx):
        e, idx = self.eng, np.asarray(idx)
        for k in ('v', 'g', 'last', 'open_', 'ring', 'ring_n'):
            setattr(e, k, np.ascontiguousarray(getattr(e, k)[idx]))
        self.w_eff, self.base = self.w_eff[idx], self.base[idx]
        if self.track:
            self.tot = self.tot[idx]
        self.B = e.B = len(idx)

    def advance(self, prob, n_steps, rec_local, bin_steps):
        from .lif_numba import _kernel
        e = self.eng
        rec_local = np.asarray(rec_local)
        lut = np.full(self.n, -1, np.int64); lut[rec_local] = np.arange(len(rec_local))
        nb = n_steps // bin_steps
        out = np.zeros((self.B, nb, len(rec_local)), np.uint16)
        inp_all = poisson_input_np(self.base, self.step_i, n_steps, prob)        # (n_steps, B, n_stim)
        dummy = np.zeros(len(self.stim_np))
        for bt in range(self.B):
            inp = np.ascontiguousarray(inp_all[:, bt, :])
            cap = max(4 * self.n, 1 << 20)
            rec_i, rec_t = np.empty(cap, np.int64), np.empty(cap, np.int64)
            done, n_rec = 0, 0
            while done < n_steps:
                k, n_rec = _kernel(n_steps - done, self.step_i + done, e.v[bt], e.g[bt], e.last[bt],
                                   e.open_[bt], e.rfc, e.indptr, e.post, e.cw, self.w_eff[bt],
                                   e.ring[bt], e.ring_n[bt], e.delay, e.stim, dummy, inp[done:], True,
                                   e.w_stim, e.v_0, e.v_rst, e.v_th, e.a, e.b, e.c, rec_i, rec_t, n_rec, -1)
                done += k
                if done < n_steps:
                    rec_i = np.concatenate([rec_i, np.empty_like(rec_i)])
                    rec_t = np.concatenate([rec_t, np.empty_like(rec_t)])
            ri, rt = rec_i[:n_rec], rec_t[:n_rec]
            col = lut[ri]; m = col >= 0
            np.add.at(out[bt], ((rt[m] - self.step_i) // bin_steps, col[m]), 1)
            if self.track:
                self.tot[bt] += np.bincount(ri, minlength=self.n)
        self.step_i += n_steps
        return out

    def totals(self):
        return self.tot


def make_engine(sub, stim_global, use_gpu=True, dtype='float32', verbose=True, torch_device=None):
    """torch on CUDA if use_gpu and a GPU is there; otherwise the numba CPU engine.
    torch_device='cpu' forces the torch engine on the CPU (slow; only for testing the GPU code path)."""
    if torch_device is not None:
        return TorchBatchLIF(sub, stim_global, torch_device, dtype, verbose=verbose)
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                return TorchBatchLIF(sub, stim_global, 'cuda', dtype, verbose=verbose)
            print("USE_GPU is on but no GPU was found -> numba on the CPU")
        except ImportError:
            print("USE_GPU is on but torch is not installed -> numba on the CPU")
    return NumbaBatchLIF(sub, stim_global, verbose=verbose)


# ---------------------------------------------------------------- brain = engine + encoder + recording
class BatchBrain:
    """Batched counterpart of interface.BrainIO: encoder rates per trial, spike counts of the
    recorded neurons in bin_ms bins, per trial."""

    def __init__(self, engine, encoder, record_global, bin_ms=5):
        self.eng, self.enc = engine, encoder
        assert np.array_equal(engine.keep_ids[engine.stim_np], encoder.neurons), \
            "engine must be built with stim_global = encoder.neurons (same order)"
        self.record = np.asarray(record_global)
        self.rec_local = np.searchsorted(engine.keep_ids, self.record)
        assert np.array_equal(engine.keep_ids[self.rec_local], self.record), "record neurons not in network"
        self.bin_ms = bin_ms
        self.bin_steps = int(round(bin_ms / engine.dt))
        self.sim_ms = 0.0                    # trial-ms simulated (sum over trials)
        self.wall_s = 0.0

    @property
    def B(self):
        return self.eng.B

    def reset(self, seeds, w_scales, track=False):
        self.eng.reset(seeds, w_scales, track)

    def keep(self, idx):
        self.eng.keep(idx)

    def rates_in(self, S):
        e = self.enc
        S = np.clip(np.atleast_2d(S), -1, 1)
        return e.r_mid + e.r_amp * e.sgn[None, :] * S[:, e.sig_of]           # (B, n_stim) Hz

    def step(self, S, n_ms):
        """Encode each trial's signals S (B, K), advance n_ms. Returns counts (B, n_bins, n_rec)."""
        t0 = time.perf_counter()
        n_steps = int(round(n_ms / self.eng.dt))
        prob = self.rates_in(S) * self.eng.dt * 1e-3
        out = self.eng.advance(prob, n_steps, self.rec_local, self.bin_steps)
        self.sim_ms += n_ms * self.eng.B
        self.wall_s += time.perf_counter() - t0
        return out


# ---------------------------------------------------------------- checks used by the notebook
def engine_check(sub, stim_global, record_global, dtype='float32', batch_sizes=(1, 8, 32, 64),
                 bench_ms=100, check_ms=200, rate=150.0, verbose=True, device='cuda'):
    """On a GPU machine: (1) correctness, torch-CUDA vs numba on identical input spikes, 4 trials of
    check_ms: share of identical spike counts per neuron, rate correlation; (2) speed, seconds of
    compute per simulated second per trial at each batch size."""
    import torch
    out = {}
    num = NumbaBatchLIF(sub, stim_global, verbose=False)
    rec = np.asarray(record_global)
    seeds, w = [11, 12, 13, 14], [1.0] * 4
    num.reset(seeds, w, track=True)
    prob = np.full((4, len(num.stim_np)), rate * 0.1e-3)
    num.advance(prob, int(check_ms / 0.1), np.searchsorted(num.keep_ids, rec), 50)
    ref = num.totals()
    del num
    for dt in sorted({'float64', dtype}):
        tb = TorchBatchLIF(sub, stim_global, device, dt, verbose=False)
        cuda = tb.dev.type == 'cuda'
        sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
        tb.reset(seeds, w, track=True)
        tb.advance(prob, int(check_ms / 0.1), np.searchsorted(tb.keep_ids, rec), 50)
        got = tb.totals().astype(np.int64)
        act = (ref + got) > 0
        out[f'match_{dt}'] = dict(identical_counts=float((ref[act] == got[act]).mean()),
                                  rate_corr=float(np.corrcoef(ref.ravel(), got.ravel())[0, 1]),
                                  spikes_numba=int(ref.sum()), spikes_torch=int(got.sum()))
        if dt == dtype:
            speed = {}
            for B in batch_sizes:
                tb.reset(list(range(B)), [1.0] * B)
                pr = np.full((B, len(tb.stim_np)), rate * 0.1e-3)
                tb.advance(pr, 20, np.arange(1), 10)                       # warm-up
                sync()
                t0 = time.perf_counter()
                tb.advance(pr, int(bench_ms / 0.1), np.searchsorted(tb.keep_ids, rec), 50)
                sync()
                s = time.perf_counter() - t0
                speed[B] = s / (bench_ms / 1000) / B
                if verbose:
                    mem = f" (GPU memory peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)" if cuda else ''
                    print(f"  batch {B:3d}: {speed[B]:.2f} s of compute per simulated second per trial{mem}")
            out['speed_s_per_trial_s'] = speed
        del tb
        if cuda:
            torch.cuda.empty_cache()
    return out
