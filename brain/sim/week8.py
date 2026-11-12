"""Week 8: make the last step (lower, let go, back off) work.

Week 7: both networks carried the cube to the zone every time but completed 0/50. The arm hovered
20-30 mm from the target holding the cube, because the week 4 expert lowers only once within 8 mm:
a sharp cut-off in a distance the brain's spiking readout can't resolve that finely (the 8 signals
are encoded on a 5 cm tanh scale, so 8 mm is a small, noisy input change).

Fixes tested here (all applied identically to every policy; the brain, its wiring and the linear
readout are unchanged):

  B  graded, press-down expert (sim/expert8.py): the height target slides from travel to placing
     height as the cube closes from 40 to 10 mm (a smooth ramp instead of an 8 mm cliff), and the
     placing target is 8 mm below the table, so dz stays clearly negative until the table stops
     the cube; the gripper opens once the cube is within 10 mm of the table.
  C  fine goal encoding: goal - cube x, y on a 2 cm tanh scale instead of 5 cm, so the last
     centimetres move the input rates 2.5x more. Same encoder neurons and channels.
  D  rate smoothing: the readout sees an exponential average of the DN rates over control steps
     (smooth = 0.5: about 2 steps), cutting spike-count noise at the cost of a little lag.

  run_network8   one network end to end with a chosen fix (encoder from its own wiring as week 7,
                 week 7's w_scale, 80 ms of brain per 40 ms step for both networks, DAgger,
                 50-seed test, feedback-cut control, video episodes)
  offline_relabel  zero-brain-time check on week 7's saved DAgger data: decode the graded expert's
                 labels from the same recorded DN rates (and smoothed rates) near the target
"""
import gc, json, pickle, time
from pathlib import Path

import numpy as np

from .pickplace import (SIGNALS8, SPEC8, ExpertPolicy, SignalPolicy, BatchBrainPolicy, dagger_pp, evaluate_pp,
                        EnvPool, PP_MAX_STEPS)
from .readout import cv_predict, r2
from .week7 import light, slug

FINE_GOAL = dict(goal_rel_x=('tanh', 0.0, 0.02), goal_rel_y=('tanh', 0.0, 0.02))

FIXES = {
    'B': dict(expert='graded', spec=None, smooth=0.0),
    'B+C': dict(expert='graded', spec=FINE_GOAL, smooth=0.0),
    'B+D': dict(expert='graded', spec=None, smooth=0.5),
    'B+C+D': dict(expert='graded', spec=FINE_GOAL, smooth=0.5),
    'week7': dict(expert='week4', spec=None, smooth=0.0),
}


def normalise_spec(obs, obs_names, spec=None, signals=SIGNALS8):
    """normalise8 with per-signal overrides (kind, centre, scale); spec=None is week 7's."""
    sp = dict(SPEC8, **(spec or {}))
    obs = np.atleast_2d(np.asarray(obs, float))
    col = {n: i for i, n in enumerate(list(obs_names))}
    out = np.empty((len(obs), len(signals)))
    for k, name in enumerate(signals):
        kind, c, sc = sp[name]
        x = (obs[:, col[name]] - c) / sc
        out[:, k] = np.clip(x, -1, 1) if kind == 'lin' else np.tanh(x)
    return out


class Week8BrainPolicy(BatchBrainPolicy):
    """BatchBrainPolicy with fix C (spec) and fix D (smooth). The rates handed to the episode log
    (and so to DAgger and the readout fit) are the smoothed ones, so training and driving see the
    same features."""

    def __init__(self, *a, spec=None, smooth=0.0, **kw):
        super().__init__(*a, **kw)
        self.spec, self.smooth = spec, float(smooth)
        self.ema = None

    def _norm(self, obs_list):
        return normalise_spec(np.array(obs_list), self.obs_names, self.spec, self.signals)

    def reset(self, seeds, obs_list):
        self.io.reset(seeds, [self.w] * len(seeds))
        self.S0 = self._norm(obs_list)
        self.io.step(self.S0, self.warmup_ms)
        self.ema = None

    def act(self, obs_list):
        S = self._norm(obs_list) if self.feedback else self.S0
        self.S = S
        self.counts = self.io.step(S, self.brain_ms)
        X = self.counts[:, -self.nb_win:, :].sum(1).astype(float) * (1000.0 / self.window_ms)
        if self.smooth:
            self.ema = X if self.ema is None else self.smooth * self.ema + (1 - self.smooth) * X
            X = self.ema.copy()
        if self.readout is None:
            A = np.tile(np.r_[0, 0, 0, 1.0], (len(X), 1))
        else:
            A = np.vstack([self.readout.predict(x[None]) for x in X[:, self.cols]])
        if self.keep_bins:
            self.last_bins = self.counts[:, :, self.pool].copy()
        return A, X

    def keep(self, idx):
        super().keep(idx)
        if self.ema is not None:
            self.ema = self.ema[np.asarray(idx)]


class SpecSignalPolicy(SignalPolicy):
    """No-brain readout on the 8 noise-free signals, with the same encoding override (fix C)."""

    def __init__(self, obs_names, kind='ridge', name=None, spec=None, **kw):
        super().__init__(obs_names, kind, name, **kw)
        self.spec = spec

    def features(self, obs):
        self.s = normalise_spec(obs, self.obs_names, self.spec, self.signals)[0]
        return self.s


def get_expert(name):
    if name == 'graded':
        from .expert8 import expert8
        return expert8
    from expert import expert
    return expert


# ---------------------------------------------------------------- no-brain references
def run_references8(env, obs_names, cfg, out, fix, resume=True):
    out = Path(out); path = out / f"week8_references_{slug(fix)}_r{len(cfg['train_seeds'])}.pkl"
    if resume and path.exists():
        print(f"  loaded {path.name}")
        return pickle.load(open(path, 'rb'))
    f = FIXES[fix]; ex = get_expert(f['expert'])
    res = {}
    outs, s = evaluate_pp(env, ExpertPolicy(ex, obs_names), ex, cfg['test_seeds'], label='expert')
    res['expert'] = dict(outs=light(outs), summary=s)
    for kind, name in (('ridge', 'linear on 8 signals'), ('mlp', 'MLP')):
        p = SpecSignalPolicy(obs_names, kind, name=name, spec=f['spec'])
        data, log = dagger_pp(env, p, ex, cfg['train_seeds'], cfg['betas'])
        outs, s = evaluate_pp(env, p, ex, cfg['test_seeds'], label=name)
        res[name] = dict(outs=light(outs), summary=s, log=log, readout=p.readout)
    pickle.dump(res, open(path, 'wb'))
    return res


# ---------------------------------------------------------------- one spiking network, end to end
def run_network8(name, sub, inputs, pool, env, obs_names, cfg, out, w_scale, fix, resume=True,
                 extra_record=(), brain_ms=80, window_ms=80):
    """Week 7's run_network with week 7's settings frozen (encoder rule, w_scale, the 80/80 timing
    both networks got in the matched-timing check) and one of FIXES applied."""
    from .interface import Encoder
    from .batch7 import make_engine, BatchBrain
    from .network7 import wiring_labels, network_stats
    out = Path(out); tag = f"{slug(name)}_{slug(fix)}"
    final = out / f'week8_{tag}.pkl'
    if resume and final.exists():
        done = pickle.load(open(final, 'rb'))
        if len(done['dagger']['log']) == len(cfg['train_seeds']):
            print(f"  loaded {final.name}")
            return done
        # more DAgger rounds requested than the saved run had: continue from the round checkpoint
        print(f"  {final.name} has {len(done['dagger']['log'])} DAgger rounds, {len(cfg['train_seeds'])} requested: "
              f"continuing from the checkpoint, then retesting")
        del done
    f = FIXES[fix]; ex = get_expert(f['expert'])
    T0 = time.perf_counter()
    pool_env = EnvPool(env)
    res = dict(name=name, fix=fix, stats={k: v for k, v in network_stats(sub).items() if k not in ('in_deg', 'out_deg')})
    labels = wiring_labels(sub, inputs, 2 * len(SIGNALS8))
    enc = Encoder(inputs, labels, SIGNALS8, kind='wiring')
    record = np.unique(np.r_[np.asarray(pool), np.asarray(extra_record, dtype=np.int64)])
    pool_cols = np.searchsorted(record, np.asarray(pool))
    eng = make_engine(sub, inputs, cfg.get('use_gpu', False), cfg.get('dtype', 'float32'),
                      torch_device=cfg.get('torch_device'))
    brain = BatchBrain(eng, enc, record)
    max_batch = cfg.get('max_batch', 64) if eng.kind == 'torch' else cfg.get('cpu_batch', 1)
    pol = Week8BrainPolicy(brain, w_scale, obs_names, pool=pool_cols, k=cfg['k'], brain_ms=brain_ms,
                           window_ms=window_ms, name=name, max_batch=max_batch, spec=f['spec'], smooth=f['smooth'])
    res.update(w_scale=w_scale, timing=[brain_ms, window_ms], pool_cols=pool_cols, record=record)
    print(f"{name}, fix {fix}: w_scale {w_scale:.3f}, {brain_ms} ms brain per step, {window_ms} ms window, "
          f"expert {f['expert']}, fine goal encoding {f['spec'] is not None}, smoothing {f['smooth']}")

    ck_npz, ck_log = out / f'week8_{tag}_dagger.npz', out / f'week8_{tag}_dagger_log.json'

    def checkpoint(r, data, log):
        np.savez(ck_npz, **{k: np.concatenate(data[k]) for k in ('X', 'Y', 'g', 'round', 'obs')})
        ck_log.write_text(json.dumps(log, default=float))

    resume_from = None
    if resume and ck_npz.exists() and ck_log.exists():
        d0 = np.load(ck_npz); resume_from = ({k: d0[k] for k in d0.files}, json.loads(ck_log.read_text()))
    t0 = time.perf_counter()
    data, log = dagger_pp(pool_env, pol, ex, cfg['train_seeds'], cfg['betas'], on_round=checkpoint,
                          resume=resume_from)
    res['dagger'] = dict(log=log, rows=int(len(data['Y'])), wall_min=(time.perf_counter() - t0) / 60,
                         alpha=pol.readout.alpha)
    res['readout'] = dict(cols=pol.cols, W=pol.readout.lin.W, alpha=pol.readout.alpha)
    del data

    sim0, wall0 = brain.sim_ms, brain.wall_s
    t0 = time.perf_counter()
    outs, s = evaluate_pp(pool_env, pol, ex, cfg['test_seeds'], label=f'{name} [{fix}]', record=True)
    res.update(outs=light(outs), summary=s, test_wall_min=(time.perf_counter() - t0) / 60)
    res['brain_s_per_sim_s'] = (brain.wall_s - wall0) / max(1e-9, (brain.sim_ms - sim0) / 1000)
    if cfg.get('cut_seeds'):
        pol.feedback = False
        co, cs = evaluate_pp(pool_env, pol, ex, cfg['cut_seeds'], label=f'{name} [{fix}], feedback cut')
        pol.feedback = True
        res['cut'] = dict(outs=light(co), summary=cs)
    order = sorted(outs, key=lambda o: (-o['success'], -o['zone'], -o['lifted'], o['seed']))
    vid = order[:cfg.get('n_video', 2)]
    fails = [o for o in outs if not o['success']]
    if fails and cfg.get('video_failure', True):
        vid.append(fails[0])
    res['video_eps'] = [dict(rec=o['rec'], seed=o['seed'], success=o['success'], outcome=o['outcome'],
                             net=f'{name} [{fix}]') for o in vid]
    del outs
    res['wall_min'] = (time.perf_counter() - T0) / 60
    pickle.dump(res, open(final, 'wb'))
    print(f"  done in {res['wall_min']:.0f} min")
    del pol, brain, eng
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return res


# ---------------------------------------------------------------- offline, zero brain time
def ema_by_episode(X, g, smooth):
    """Exponential smoothing of rows within each episode (rows of an episode are contiguous)."""
    if not smooth:
        return X
    Y = np.empty_like(X, dtype=float)
    first = np.r_[True, g[1:] != g[:-1]]
    acc = None
    for i in range(len(X)):
        acc = X[i].astype(float) if first[i] else smooth * acc + (1 - smooth) * X[i]
        Y[i] = acc
    return Y


def offline_relabel(npz_path, pool_cols, obs_names, near_mm=50, smooths=(0.0, 0.5)):
    """Week 7's saved DAgger data (DN rates at every visited state) relabelled with the graded
    expert. Out-of-episode ridge decoding of each expert's labels from the same rates, on all rows
    and on the states that matter now: holding the cube within near_mm of the target.
    Reported: dz R2, and how often the decoded dz says 'go down' (< -2 mm/step) where the label
    does, and 'stay' where the label says stay. Costs no brain time."""
    from expert import expert as e4
    from .expert8 import expert8 as e8
    d = np.load(npz_path)
    X, g, obs = d['X'][:, pool_cols].astype(float), d['g'], d['obs']
    Y4 = np.array([e4(o)[0] for o in obs]); Y8 = np.array([e8(o)[0] for o in obs])
    held = (obs[:, 10] > 0.045)
    near = held & (np.hypot(obs[:, 19], obs[:, 20]) < near_mm / 1000)
    rows = []
    for sm in smooths:
        Xs = ema_by_episode(X, g, sm)
        for lab, Y in (('week 4 expert (8 mm cut-off)', Y4), ('graded, press-down expert', Y8)):
            P, _ = cv_predict(Xs, Y[:, :3], g)
            dz, pz = Y[near, 2], P[near, 2]
            down = dz < -0.002
            rows.append({'labels': lab, 'smoothing': sm,
                         'dz R2, all states': round(float(r2(Y[:, 2], P[:, 2])[0]), 3),
                         'dz R2, holding near target': round(float(r2(dz, pz)[0]), 3) if near.sum() > 10 else None,
                         "label says 'go down' near target": f"{down.mean():.0%}",
                         'decoded agrees when label says down': f"{(pz[down] < -0.002).mean():.0%}" if down.any() else '-',
                         'decoded agrees when label says stay': f"{(pz[~down] >= -0.002).mean():.0%}" if (~down).any() else '-',
                         'rows near target': int(near.sum())})
    return rows
