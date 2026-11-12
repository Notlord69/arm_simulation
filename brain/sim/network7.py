"""Week 7 network helpers: the random baseline network and the rules both networks share.

  random_network(sub, seed)        degree-, weight- and sign-preserving rewiring (week 2/5 shuffle_sub)
  wiring_labels(sub, inputs, k)    week 5's 'wiring' encoder channels, computed on any network's own
                                   connections (so the random network gets the same rule, not the
                                   connectome's channels)
  frac_firing / match_w_scale      week 5's activity-matching rule for the random network's w_scale
  network_stats                    checks that the random network matches on size, degrees and E/I
"""
import numpy as np
import scipy.sparse as sp

from .lif import shuffle_sub


def random_network(sub, seed=0):
    """Same neurons; the post-synaptic end of every connection permuted across all connections.
    Keeps: neuron count, every neuron's in-degree and out-degree, every connection's signed
    synapse count (sign set by the presynaptic neuron, so E/I is unchanged)."""
    return shuffle_sub(sub, seed=seed)


def input_rows(sub, inputs_global):
    """Unsigned synapse counts of the input neurons' outgoing connections only:
    rows = the input neurons (in the given order), cols = local post index. Built from the
    input neurons' edges alone, so it stays small on the full network."""
    keep = np.asarray(sub['keep'])
    n = len(keep)
    rows = np.searchsorted(keep, np.asarray(inputs_global))
    assert np.array_equal(keep[rows], np.asarray(inputs_global)), "inputs must be in sub['keep']"
    pre = np.asarray(sub['pre'])
    lut = np.full(n, -1, np.int64); lut[rows] = np.arange(len(rows))
    m = lut[pre] >= 0
    return sp.csr_matrix((np.abs(np.asarray(sub['w'])[m]).astype(float),
                          (lut[pre[m]], np.asarray(sub['post'])[m])), shape=(len(rows), n))


def wiring_labels(sub, inputs_global, n_channels):
    """The week 5 'wiring' channel split (interface.wiring_channels), on this network's wiring:
    each input neuron's row of output synapses, log1p, unit norm; split recursively in half along
    the first principal component. n_channels must be a power of two."""
    depth = int(round(np.log2(n_channels)))
    assert 2 ** depth == n_channels, "n_channels must be a power of two"
    A = input_rows(sub, inputs_global)
    rows = np.arange(A.shape[0])
    cols = np.unique(A.indices)
    P = np.log1p(A[:, cols].toarray())
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12

    def split(idx, d):
        if d == 0:
            return [idx]
        X = P[idx] - P[idx].mean(0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        score = X @ vt[0]
        if vt[0][np.argmax(np.abs(vt[0]))] < 0:
            score = -score
        order = idx[np.argsort(score, kind='stable')]
        lo, hi = np.array_split(order, 2)
        return split(lo, d - 1) + split(hi, d - 1)

    lab = np.empty(len(rows), int)
    for c, part in enumerate(split(np.arange(len(rows)), depth)):
        lab[part] = c
    return lab


def channel_overlap_sub(sub, inputs_global, labels):
    """Cosine similarity between channels' summed output rows (week 5's channel_overlap)."""
    A = input_rows(sub, inputs_global)
    k = labels.max() + 1
    R = np.vstack([np.asarray(A[labels == c].sum(0)).ravel() for c in range(k)])
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
    return R @ R.T


def network_stats(sub):
    """Size, degree and E/I summary, to show the random network matches the connectome.
    in_deg / out_deg: each neuron's number of connections in and out (compare per neuron with
    degrees_equal). exc_*: excitatory share of neurons, connections and synapses."""
    n = len(sub['keep'])
    pre, post, w = np.asarray(sub['pre']), np.asarray(sub['post']), np.asarray(sub['w'])
    kout, kin = np.bincount(pre, minlength=n), np.bincount(post, minlength=n)
    exc_neuron = np.zeros(n, bool); inh_neuron = np.zeros(n, bool)
    exc_neuron[pre[w > 0]] = True; inh_neuron[pre[w < 0]] = True
    return dict(neurons=n, connections=len(w), synapses=int(np.abs(w).sum()),
                exc_neurons=int(exc_neuron.sum()), inh_neurons=int(inh_neuron.sum()),
                exc_conn_frac=float((w > 0).mean()), exc_syn_frac=float(w[w > 0].sum() / np.abs(w).sum()),
                in_deg_mean=float(kin.mean()), in_deg_median=float(np.median(kin)), in_deg_max=int(kin.max()),
                out_deg_median=float(np.median(kout)), out_deg_max=int(kout.max()),
                self_loops=int((pre == post).sum()), in_deg=kin, out_deg=kout)


def degrees_equal(a, b):
    """True if every neuron has the same in- and out-degree in both networks."""
    return bool(np.array_equal(a['in_deg'], b['in_deg']) and np.array_equal(a['out_deg'], b['out_deg']))


def frac_firing(io, w_scale, states, others_local, seed=99, warmup_ms=200):
    """Share of non-input neurons firing >= 1 Hz while the brain is driven through `states`
    (one 40 ms step each), from a fresh brain held warmup_ms at the first state."""
    before = io.neuron_counts.copy(); ms0 = io.sim_ms
    io.reset(seed, w_scale)
    io.step(states[0], warmup_ms)
    for s in states:
        io.step(s, 40)
    hz = (io.neuron_counts - before) / ((io.sim_ms - ms0) / 1000)
    return float((hz[others_local] >= 1).mean()), hz


def match_w_scale(io, w_ref, target, states, others_local, n_iter=8, verbose=True):
    """Week 5's rule: bisect (in log space) on w_scale until the share of non-input neurons
    firing >= 1 Hz matches `target`, starting from the connectome's w_ref."""
    f0, _ = frac_firing(io, w_ref, states, others_local)
    lo, hi = (0.02 * w_ref, w_ref) if f0 > target else (w_ref, 4 * w_ref)
    if verbose:
        print(f"  at w_scale {w_ref:.3f}: {f0:.2%} firing (target {target:.2%})")
    for _ in range(n_iter):
        mid = float(np.sqrt(lo * hi))
        f, _ = frac_firing(io, mid, states, others_local)
        lo, hi = (mid, hi) if f < target else (lo, mid)
        if verbose:
            print(f"  try {mid:.3f}: {f:.2%}")
    return float(np.sqrt(lo * hi))
