"""Linear readouts with episode-grouped cross-validation (week 5 analysis, reused in week 6).

Everything is plain numpy so it runs in sim-env without scikit-learn.

Rules that keep the numbers honest:
  * folds are split by whole episode, never by row (neighbouring 40 ms steps are nearly
    identical, so a row split would leak);
  * the ridge penalty is chosen inside each training fold (nested CV);
  * features are standardised with training-fold statistics only.
"""
import numpy as np

ALPHAS = np.logspace(-2, 5, 15)


def _folds(groups, n_folds, seed):
    g = np.unique(groups)
    g = np.random.default_rng(seed).permutation(g)
    return [np.isin(groups, part) for part in np.array_split(g, n_folds)]


def _fit_path(X, Y, alphas):
    """Ridge coefficients for every alpha at once (X, Y already centred)."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    UtY = U.T @ Y
    return [Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY) for a in alphas]


def _standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    ok = sd > 1e-9
    return (Xtr[:, ok] - mu[ok]) / sd[ok], (Xte[:, ok] - mu[ok]) / sd[ok]


def ridge_fit_predict(Xtr, Ytr, Xte, gtr, alphas=ALPHAS, inner=3, seed=0):
    """Fit on (Xtr, Ytr), choosing alpha by inner CV grouped by gtr; predict Xte.
    One alpha is chosen jointly for all target columns."""
    Xtr, Xte = _standardise(Xtr, Xte)
    ym = Ytr.mean(0)
    if len(np.unique(gtr)) >= inner and len(alphas) > 1:
        err = np.zeros(len(alphas))
        for f in _folds(gtr, inner, seed + 1):
            a, b = _standardise(Xtr[~f], Xtr[f])
            m = Ytr[~f].mean(0)
            for i, W in enumerate(_fit_path(a, Ytr[~f] - m, alphas)):
                err[i] += ((b @ W + m - Ytr[f]) ** 2).sum()
        alpha = alphas[int(np.argmin(err))]
    else:
        alpha = alphas[len(alphas) // 2]
    W = _fit_path(Xtr, Ytr - ym, [alpha])[0]
    return Xte @ W + ym, alpha


def cv_predict(X, Y, groups, n_folds=5, seed=0, alphas=ALPHAS):
    """Out-of-fold predictions for every row (each row predicted by a model that never
    saw its episode). Returns (pred, chosen alpha per fold)."""
    Y2 = Y.reshape(len(Y), -1).astype(float)
    pred = np.zeros_like(Y2)
    chosen = []
    for f in _folds(groups, n_folds, seed):
        pred[f], a = ridge_fit_predict(X[~f], Y2[~f], X[f], groups[~f], alphas, seed=seed)
        chosen.append(a)
    return pred.reshape(Y.shape), chosen


def r2(Y, P):
    """Coefficient of determination per column (0 = no better than predicting the mean)."""
    Y = Y.reshape(len(Y), -1); P = P.reshape(len(P), -1)
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def cv_r2(X, Y, groups, **kw):
    return r2(Y, cv_predict(X, Y, groups, **kw)[0])


def cv_classify(X, labels, groups, **kw):
    """Ridge classifier (one-vs-rest regression on one-hot labels, argmax).
    Returns (balanced accuracy, chance level, out-of-fold predicted labels)."""
    classes = np.unique(labels)
    onehot = (labels[:, None] == classes[None]).astype(float)
    P, _ = cv_predict(X, onehot, groups, **kw)
    pred = classes[P.argmax(1)]
    bal = np.mean([(pred[labels == c] == c).mean() for c in classes])
    return bal, 1 / len(classes), pred


def episode_swap(Y, episode, step, rng):
    """Null labels: each episode gets the labels of a different, randomly chosen episode
    at the SAME step number. Rows past the shorter episode are dropped. This keeps
    everything the episodes share over time (all start with a reach, the brain's own
    warm-up drift, ...), so beating it means the readout tracks THIS episode's state.
    Returns (row mask, swapped labels for the kept rows)."""
    eps = np.unique(episode)
    perm = rng.permutation(eps)
    while len(eps) > 1 and np.any(perm == eps):              # no episode keeps its own labels
        perm = rng.permutation(eps)
    src = dict(zip(eps, perm))
    index = {(e, t): i for i, (e, t) in enumerate(zip(episode, step))}
    keep = np.zeros(len(Y), bool)
    newY = np.zeros_like(Y)
    for i, (e, t) in enumerate(zip(episode, step)):
        j = index.get((src[e], t))
        if j is not None:
            keep[i] = True
            newY[i] = Y[j]
    return keep, newY[keep]


def null_r2(X, Y, episode, step, n_perm=20, seed=0, **kw):
    """R^2 (per target column) of readouts fit to episode-swapped labels, n_perm times."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        keep, Ys = episode_swap(Y, episode, step, rng)
        out.append(cv_r2(X[keep], Ys, episode[keep], **kw))
    return np.array(out)


def lagged(X, Y, episode, step, lag):
    """Pair brain window at step t with the state at step t - lag (same episode).
    lag > 0: the brain reports the past (it lags). lag < 0: it would be predicting
    the future, which an open-loop brain cannot do beyond what the state's smoothness allows."""
    index = {(e, t): i for i, (e, t) in enumerate(zip(episode, step))}
    xi, yi = [], []
    for i, (e, t) in enumerate(zip(episode, step)):
        j = index.get((e, t - lag))
        if j is not None:
            xi.append(i); yi.append(j)
    xi, yi = np.array(xi), np.array(yi)
    return X[xi], Y[yi], episode[xi]
