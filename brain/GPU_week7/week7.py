"""Week 7 orchestration. Everything the notebook runs is here as functions, so exactly this code
was tested in the sandbox (on a stand-in network) and runs on the laptop (on the male CNS).

  run_references   expert; linear readout and MLP on the 8 noise-free signals (physics only)
  run_network      one spiking network end to end: encoder from its own wiring, w_scale,
                   encoder tuning sweep, timing sweep, DAgger, 50-seed test, video episodes
  offline_compare  open-loop decoding on round 0, where every network saw IDENTICAL states
  comparison_table / verdict   the deliverable and the pre-registered decision

Every stage writes its result to OUT, and RESUME = True skips what is already there, so the
run can be stopped and restarted (DAgger resumes after the last finished round).

Engines (cfg['use_gpu'], cfg['dtype'], cfg['max_batch']): on a GPU every stage runs its episodes
side by side (sim/batch7.py); on the CPU (numba) they run one after another, as in week 6.
"""
import gc, json, pickle, time
from pathlib import Path

import numpy as np

from .interface import Encoder
from .batch7 import make_engine, BatchBrain
from .readout import cv_r2, cv_predict, ridge_fit_predict, r2, _folds
from .pickplace import (SIGNALS8, ACT4, normalise8, ExpertPolicy, SignalPolicy, BatchBrainPolicy, run_pp,
                        dagger_pp, evaluate_pp, summarise_pp, paired_diff, paired_milestones,
                        timing_sweep, choose_timing, informative, EnvPool)
from .network7 import wiring_labels, channel_overlap_sub, network_stats

HEAVY = ('X', 'bins', 'rec')


def light(outs):
    return [{k: v for k, v in o.items() if k not in HEAVY} for o in outs]


def slug(name):
    return ''.join(c if c.isalnum() else '_' for c in name.lower()).strip('_')


def reference_states(env, expert, obs_names, seed=99_999, n=17):
    """The first 0.68 s of an expert episode, as encoded signals (week 5's activity-matching input)."""
    o = env.reset(seed); S = []
    for _ in range(n):
        S.append(normalise8(o, obs_names)[0])
        o = env.step(expert(o)[0])
    return np.array(S)


# ---------------------------------------------------------------- no-brain policies
def run_references(env, expert, obs_names, cfg, out, resume=True):
    path = Path(out) / 'week7_references.pkl'
    if resume and path.exists():
        print(f"  loaded {path.name}")
        return pickle.load(open(path, 'rb'))
    res = {}
    outs, s = evaluate_pp(env, ExpertPolicy(expert, obs_names), expert, cfg['test_seeds'], label='expert')
    res['expert'] = dict(outs=light(outs), summary=s)
    for kind, name in (('ridge', 'linear on 8 signals'), ('mlp', 'MLP')):
        p = SignalPolicy(obs_names, kind, name=name)
        t0 = time.perf_counter()
        data, log = dagger_pp(env, p, expert, cfg['train_seeds'], cfg['betas'])
        outs, s = evaluate_pp(env, p, expert, cfg['test_seeds'], label=name)
        res[name] = dict(outs=light(outs), summary=s, log=log, rows=int(len(data['Y'])),
                         readout=p.readout, wall_min=(time.perf_counter() - t0) / 60)
    pickle.dump(res, open(path, 'wb'))
    return res


# ---------------------------------------------------------------- activity and encoder checks (batched)
def frac_firing_batch(brain, w_list, states, others_local, seed=99, warmup_ms=200):
    """Share of non-input neurons firing >= 1 Hz, one trial per w_scale in w_list, all driven by
    the SAME input (same seed, the reference states, one 40 ms step each)."""
    B = len(w_list)
    brain.reset([seed] * B, list(w_list), track=True)
    brain.step(np.tile(states[0], (B, 1)), warmup_ms)
    for st in states:
        brain.step(np.tile(st, (B, 1)), 40)
    hz = brain.eng.totals() / ((warmup_ms + 40 * len(states)) / 1000)
    return (hz[:, others_local] >= 1).mean(1), hz


def match_w_scale_batch(brain, w_ref, target, states, others_local, grid=12, max_rounds=4, verbose=True):
    """Week 5's rule (match the share of non-input neurons firing), as a batched search: a grid of
    `grid` w_scales from w_ref/4 to 4 w_ref, then a second grid inside the bracketing pair."""
    lo, hi, tries, refined = w_ref / 4, w_ref * 4, [], 0
    for _ in range(max_rounds):
        ws = np.geomspace(lo, hi, grid)
        f, _ = frac_firing_batch(brain, ws, states, others_local)
        tries += list(zip(ws.tolist(), f.tolist()))
        above = np.flatnonzero(f >= target)
        if len(above) == 0:
            lo, hi = ws[-1], ws[-1] * 4
        elif above[0] == 0:
            lo, hi = ws[0] / 4, ws[0]
        else:
            lo, hi = ws[above[0] - 1], ws[above[0]]
            refined += 1
            if refined == 2:
                break
        if verbose:
            print(f"  grid {ws[0]:.3f}-{ws[-1]:.3f}: firing {f.min():.1%}-{f.max():.1%} (target {target:.1%}) "
                  f"-> next {lo:.3f}-{hi:.3f}")
    return float(np.sqrt(lo * hi)), tries


def sweep_r2(brain, w_scale, cols, seeds=(0, 1), levels=np.linspace(-1, 1, 9), hold_ms=300, settle_ms=100,
             warmup_ms=200):
    """Week 5's tuning-sweep check for the 8-signal encoder, all trials side by side: each signal
    swept alone over 9 levels (random order, 300 ms each, the others at 0) from a fresh brain, for
    2 seeds. A ridge readout from `cols` fitted on one seed's sweep is scored on the other's (both
    ways). Returns held-out R2 per signal."""
    K = brain.enc.K
    trials = [(sd, k) for sd in seeds for k in range(K)]
    perms = {}
    for sd in seeds:
        rng = np.random.default_rng(sd)
        for k in range(K):
            perms[sd, k] = rng.permutation(levels)
    B = len(trials)
    brain.reset([sd * 1000 + k for sd, k in trials], [w_scale] * B)
    brain.step(np.zeros((B, K)), warmup_ms)
    nb_settle, nb_hold, nb_w = (int(round(x / brain.bin_ms)) for x in (settle_ms, hold_ms, 40))
    rows = {sd: [] for sd in seeds}
    for li in range(len(levels)):
        S = np.zeros((B, K))
        for b, (sd, k) in enumerate(trials):
            S[b, k] = perms[sd, k][li]
        c = brain.step(S, hold_ms)
        for b, (sd, k) in enumerate(trials):
            for w0 in range(nb_settle, nb_hold - nb_w + 1, nb_w):
                rows[sd].append((k, S[b, k], c[b, w0:w0 + nb_w].sum(0)))

    def xy(rr):
        X = np.array([r[2][cols] for r in rr], float) * (1000.0 / 40)
        Y = np.zeros((len(rr), K)); g = np.zeros(len(rr), int)
        for i, (k, lv, _) in enumerate(rr):
            Y[i, k] = lv; g[i] = k * 100 + int(round(lv * 4))
        return X, Y, g
    sc = []
    for a, b in ((seeds[0], seeds[1]), (seeds[1], seeds[0])):
        Xa, Ya, ga = xy(rows[a]); Xb, Yb, _ = xy(rows[b])
        P, _ = ridge_fit_predict(Xa, Ya, Xb, ga)
        sc.append(r2(Yb, P))
    return dict(zip(SIGNALS8, np.mean(sc, 0).round(3).tolist()))


# ---------------------------------------------------------------- one spiking network, end to end
def run_network(name, sub, inputs, pool, env, expert, obs_names, cfg, out, w_scale=None,
                match_target=None, ref_states=None, resume=True, extra_record=()):
    """name: 'connectome' or 'random network'. sub: the network (WeightSet.subnetwork format).
    inputs: global ids of the JO-C/E input neurons. pool: global ids of the readout pool (all DNs).
    w_scale: a number (connectome), or None to match match_target (share of non-input neurons
    firing >= 1 Hz on ref_states), the week 5 rule for the random network."""
    out = Path(out); tag = slug(name)
    final = out / f'week7_{tag}.pkl'
    if resume and final.exists():
        print(f"  loaded {final.name}")
        return pickle.load(open(final, 'rb'))
    T0 = time.perf_counter()
    pool_env = EnvPool(env)
    res = dict(name=name, stats=network_stats(sub))
    print(f"{name}: {res['stats']['neurons']:,} neurons, {res['stats']['connections']:,} connections")

    # encoder: week 5's wiring split, on THIS network's connections, 16 channels = 8 signals x push/pull
    labels = wiring_labels(sub, inputs, 2 * len(SIGNALS8))
    ov = channel_overlap_sub(sub, inputs, labels)
    res['encoder_overlap'] = float(ov[~np.eye(len(ov), dtype=bool)].mean())
    enc = Encoder(inputs, labels, SIGNALS8, kind='wiring')
    enc.to_json(out / f'encoder8_{tag}.json')
    record = np.unique(np.r_[np.asarray(pool), np.asarray(extra_record, dtype=np.int64)])
    pool_cols = np.searchsorted(record, np.asarray(pool))
    res['record'], res['pool_cols'] = record, pool_cols
    eng = make_engine(sub, inputs, cfg.get('use_gpu', False), cfg.get('dtype', 'float32'),
                      torch_device=cfg.get('torch_device'))
    brain = BatchBrain(eng, enc, record)
    res['engine'] = f"{eng.kind}" + (f" {cfg.get('dtype', 'float32')} on {eng.dev}" if eng.kind == 'torch' else ' (CPU)')
    max_batch = cfg.get('max_batch', 64) if eng.kind == 'torch' else cfg.get('cpu_batch', 1)
    keep = np.asarray(sub['keep'])
    others = np.setdiff1d(np.arange(len(keep)), np.searchsorted(keep, inputs))
    pool_local = np.searchsorted(keep, pool)
    print(f"  engine: {res['engine']}, up to {max_batch} episodes side by side; encoder: 16 channels of about "
          f"{len(inputs) // 16} input neurons, output overlap between channels {res['encoder_overlap']:.2f}")

    # w_scale
    meta_path = out / f'week7_{tag}_meta.json'
    meta = json.loads(meta_path.read_text()) if (resume and meta_path.exists()) else {}
    if 'w_scale' in meta:
        w = meta['w_scale']
    elif w_scale is not None:
        w = float(w_scale)
    else:
        print(f"  matching activity: target {match_target:.2%} of non-input neurons firing >= 1 Hz")
        w, meta['w_tries'] = match_w_scale_batch(brain, cfg['w_ref'], match_target, ref_states, others)
    f, hz = frac_firing_batch(brain, [w], ref_states, others)
    res.update(w_scale=w, frac_firing=float(f[0]), pool_mean_hz=float(hz[0, pool_local].mean()),
               pool_active=float((hz[0, pool_local] >= 1).mean()))
    print(f"  w_scale {w:.3f}: {res['frac_firing']:.2%} of non-input neurons fire >= 1 Hz; readout pool "
          f"{len(pool)} neurons, mean {res['pool_mean_hz']:.1f} Hz, {res['pool_active']:.0%} active")
    meta['w_scale'] = w
    meta_path.write_text(json.dumps(meta))

    # encoder tuning sweep (diagnostic)
    if cfg.get('sweep', True):
        if 'sweep_r2' not in meta:
            t0 = time.perf_counter()
            meta['sweep_r2'] = sweep_r2(brain, w, pool_cols)
            meta_path.write_text(json.dumps(meta))
            print(f"  encoder sweep ({time.perf_counter() - t0:.0f} s): held-out R2 per signal {meta['sweep_r2']}")
        res['sweep_r2'] = meta['sweep_r2']

    mk = lambda bms, wms=None: BatchBrainPolicy(brain, w, obs_names, pool=pool_cols, k=cfg['k'], brain_ms=bms,
                                               window_ms=wms or min(40, bms), name=name, max_batch=max_batch)
    # timing: brain ms per control step and decoder window, by choose_timing's fixed rule
    if 'timing' not in meta:
        rows, _ = timing_sweep(pool_env, mk, expert, cfg['timing_seeds'], cfg['timing_settings'])
        meta['timing'] = dict(rows=rows, choice=list(choose_timing(rows)))
        meta_path.write_text(json.dumps(meta, default=float))
    res['timing'] = meta['timing']
    brain_ms, window_ms = res['timing']['choice']
    print(f"  timing: {brain_ms} ms of brain per 40 ms control step, {window_ms} ms decoder window")

    # DAgger, with a checkpoint after every round
    pol = mk(brain_ms, window_ms)
    ck_npz, ck_log = out / f'week7_{tag}_dagger.npz', out / f'week7_{tag}_dagger_log.json'

    def checkpoint(r, data, log):
        np.savez(ck_npz, **{k: np.concatenate(data[k]) for k in ('X', 'Y', 'g', 'round', 'obs')})
        ck_log.write_text(json.dumps(log, default=float))

    resume_from = None
    if resume and ck_npz.exists() and ck_log.exists():
        d0 = np.load(ck_npz); resume_from = ({k: d0[k] for k in d0.files}, json.loads(ck_log.read_text()))
    brain.sim_ms = brain.wall_s = 0.0
    t0 = time.perf_counter()
    data, log = dagger_pp(pool_env, pol, expert, cfg['train_seeds'], cfg['betas'], on_round=checkpoint,
                          resume=resume_from)
    res['dagger'] = dict(log=log, rows=int(len(data['Y'])), wall_min=(time.perf_counter() - t0) / 60,
                         alpha=pol.readout.alpha, k=len(pol.cols))
    m0 = data['round'] == 0
    res['round0'] = dict(X=data['X'][m0][:, pool_cols].astype(np.float32), Y=data['Y'][m0], g=data['g'][m0],
                         S=normalise8(data['obs'][m0], obs_names))
    res['readout'] = dict(cols=pol.cols, W=pol.readout.lin.W, alpha=pol.readout.alpha)
    del data

    # test (recorded, so the video uses the test episodes themselves)
    t0 = time.perf_counter()
    sim0, wall0 = brain.sim_ms, brain.wall_s
    outs, s = evaluate_pp(pool_env, pol, expert, cfg['test_seeds'], label=name, record=True)
    res.update(outs=light(outs), summary=s, test_wall_min=(time.perf_counter() - t0) / 60)
    res['brain_s_per_sim_s'] = (brain.wall_s - wall0) / max(1e-9, (brain.sim_ms - sim0) / 1000)

    # control: the same trained network and readout, but the brain only ever sees the first state
    if cfg.get('cut_seeds'):
        pol.feedback = False
        co, cs = evaluate_pp(pool_env, pol, expert, cfg['cut_seeds'], label=f'{name}, feedback cut')
        pol.feedback = True
        res['cut'] = dict(outs=light(co), summary=cs)

    # episodes for the video: the best two test episodes (success first, then how far they got)
    order = sorted(outs, key=lambda o: (-o['success'], -o['zone'], -o['lifted'], o['seed']))
    res['video_eps'] = [dict(rec=o['rec'], seed=o['seed'], success=o['success'], outcome=o['outcome'], net=name)
                        for o in order[:cfg.get('n_video', 2)]]
    del outs
    res['brain_s'] = brain.sim_ms / 1000
    res['wall_min'] = (time.perf_counter() - T0) / 60
    pickle.dump(res, open(final, 'wb'))
    print(f"  {name} done in {res['wall_min']:.0f} min; brain compute {res['brain_s_per_sim_s']:.2f} s "
          f"per simulated second per episode in the test")
    del pol, brain, eng
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return res


def signal_video_eps(env, expert, obs_names, readout, name, seeds):
    p = SignalPolicy(obs_names, 'mlp', name=name); p.readout = readout
    vid = []
    for sd in seeds:
        e = run_pp(env, p, expert, sd, record=True)
        vid.append(dict(rec=e['rec'], seed=sd, success=e['success'], outcome=e['outcome'], net=name))
    return vid


# ---------------------------------------------------------------- open-loop comparison on identical states
def _cv_select_r2(X, Y, g, k, n_folds=5, seed=0):
    """CV R2 (dx, dy, dz) and grip balanced accuracy, with the k-most-informative selection made
    inside each training fold (no leakage). k=None uses every column."""
    P = np.zeros_like(Y)
    for f in _folds(g, n_folds, seed):
        cols = np.arange(X.shape[1]) if k is None else informative(X[~f], Y[~f], k)[0]
        P[f], _ = ridge_fit_predict(X[~f][:, cols], Y[~f], X[f][:, cols], g[~f])
    lab = Y[:, 3] > 0
    bal = 0.5 * (np.mean((P[:, 3] > 0)[lab]) + np.mean((P[:, 3] <= 0)[~lab]))
    return dict(zip(('dx', 'dy', 'dz'), r2(Y[:, :3], P[:, :3]).round(3).tolist()), grip_bal_acc=round(float(bal), 3))


def offline_compare(nets, k_small=68):
    """Round 0: the expert drives with the same seeds and noise, so every network was driven
    through IDENTICAL states (checked here). Decoding there is a clean open-loop comparison:
    expert action (all DNs, and the k most informative), and the 8 encoded signals."""
    names = list(nets)
    Y0 = nets[names[0]]['round0']['Y']
    same = all(np.array_equal(nets[n]['round0']['Y'], Y0) for n in names)
    rows = {}
    for n in names:
        r0 = nets[n]['round0']
        X, Y, g, S = r0['X'].astype(float), r0['Y'], r0['g'], r0['S']
        rows[n] = dict(action_all=_cv_select_r2(X, Y, g, None),
                       action_k=_cv_select_r2(X, Y, g, k_small),
                       signals=dict(zip(SIGNALS8, cv_r2(X, S, g).round(3).tolist())))
    return rows, same


# ---------------------------------------------------------------- the deliverable
def comparison_rows(results):
    """results: {label: summary dict from summarise_pp}. Returns a list of table rows."""
    rows = []
    for k, s in results.items():
        rows.append({'policy': k, 'success': f"{s['success']}/{s['n']}",
                     'success rate': s['success_rate'], '95% CI': f"{s['ci'][0]:.0%}-{s['ci'][1]:.0%}",
                     'placement error, successes (mm, median)': s['place_mm_success'],
                     'placement error, all lifted (mm, median)': s['place_mm_lifted'],
                     'time to complete (s, median)': s['time_s'],
                     'lifted': s['lifted_rate'], 'carried to zone': s['zone_rate'],
                     'stalled': s['stalled_rate']})
    return rows


def verdict(conn_outs, rand_outs, alpha=0.05):
    """Pre-registered: compare the connectome with the random network on the same 50 seeds,
    milestone by milestone, from the hardest (success) to the easiest (lifted). The answer is
    the first milestone with a significant paired difference (exact McNemar, p < alpha) in either
    direction. Placement accuracy and time are compared only if both succeed on >= 10 seeds."""
    pm = paired_milestones(conn_outs, rand_outs)
    for m in ('success', 'zone', 'lifted'):
        d = pm[m]
        if d['p'] < alpha:
            better = 'connectome' if d['only_a'] > d['only_b'] else 'random network'
            label = {'success': 'completing the task', 'zone': 'carrying the cube to the zone',
                     'lifted': 'lifting the cube'}[m]
            return (f"{better} is better at {label}: {d['a']:.0%} vs {d['b']:.0%} "
                    f"(seeds only the connectome did: {d['only_a']}, only the random network: "
                    f"{d['only_b']}, McNemar p = {d['p']:.3g})"), pm
    return ("no significant difference between the connectome and the random network on any "
            "milestone at n = %d" % len(conn_outs)), pm
