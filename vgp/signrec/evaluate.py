"""Cross-validation for the DTW k-NN classifier."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold

from .data import Clip, FeatureConfig, extract
from .dtw import DTWKNN


@dataclass
class FoldResult:
    test_idx: np.ndarray
    dists: np.ndarray          # (n_test, n_train)
    model: DTWKNN


def splits_loro(clips: list[Clip]):
    rounds = np.array([c.round for c in clips])
    for r in sorted(set(rounds)):
        yield f"round{r}", np.where(rounds != r)[0], np.where(rounds == r)[0]


def splits_kfold(clips: list[Clip], n_splits=5, n_repeats=5, seed=0):
    y = np.array([c.label for c in clips])
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for i, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), y)):
        yield f"rep{i // n_splits}fold{i % n_splits}", tr, te


def session_normalise(feats, clips):
    """Z-score each recording round with its own (label-free) statistics.

    Simulates a per-session calibration: live, the stats would come from a short
    unlabelled recording at the start of the session.
    """
    from .data import Standardizer
    rounds = np.array([c.round for c in clips])
    out = list(feats)
    for r in set(rounds):
        idx = np.where(rounds == r)[0]
        sc = Standardizer().fit([feats[i] for i in idx])
        for i in idx:
            out[i] = sc.transform(feats[i])
    return out


def run_folds(clips, feats, splits, band) -> list[FoldResult]:
    labels = [c.label for c in clips]
    out = []
    for _, tr, te in splits:
        m = DTWKNN(k=1, band=band).fit_features([feats[i] for i in tr], [labels[i] for i in tr])
        out.append(FoldResult(te, m.distances_features([feats[i] for i in te]), m))
    return out


def score(clips, folds: list[FoldResult], k: int, n_groups: int):
    """Accuracy per group of folds (a repeat for k-fold, a round for LORO) and pooled predictions."""
    y = np.array([c.label for c in clips])
    per_fold_correct, preds, nearest = [], [], []
    for f in folds:
        f.model.k = k
        dec = f.model.decide(f.dists)
        p = np.array([d["best_label"] for d in dec])
        per_fold_correct.append((p == y[f.test_idx]))
        preds.append((f.test_idx, p, np.array([d["nearest_distance"] for d in dec])))
    # group folds: LORO -> 1 fold per group; k-fold -> n_splits folds per repeat
    per = len(folds) // n_groups
    accs = [np.concatenate(per_fold_correct[g * per:(g + 1) * per]).mean() for g in range(n_groups)]
    return np.array(accs), preds


def confusion(clips, preds, classes):
    y = np.array([c.label for c in clips])
    yt = np.concatenate([y[i] for i, _, _ in preds])
    yp = np.concatenate([p for _, p, _ in preds])
    return confusion_matrix(yt, yp, labels=classes)


def grid(clips, configs: list[FeatureConfig], ks=(1, 3), band=0.15, session_norm=False):
    rows, details = [], {}
    for cfg in configs:
        t0 = time.time()
        feats = [extract(c.df, cfg) for c in clips]
        if session_norm:
            feats = session_normalise(feats, clips)
        loro = run_folds(clips, feats, splits_loro(clips), band)
        kf = run_folds(clips, feats, splits_kfold(clips), band)
        elapsed = time.time() - t0
        for k in ks:
            a_loro, p_loro = score(clips, loro, k, n_groups=3)
            a_kf, p_kf = score(clips, kf, k, n_groups=5)
            name = f"dtw-{k}nn  {cfg.name()}" + ("  +session-norm" if session_norm else "")
            rows.append({"model": name, "loro_mean": a_loro.mean(), "loro_per_round": a_loro.round(3).tolist(),
                         "kfold_mean": a_kf.mean(), "kfold_std": a_kf.std(), "seconds": round(elapsed, 1)})
            details[name] = {"cfg": cfg, "k": k, "session_norm": session_norm,
                             "loro_preds": p_loro, "kfold_preds": p_kf}
        print(f"  {cfg.name():32s} done in {elapsed:.1f}s", flush=True)
    return rows, details


# -- reject threshold -------------------------------------------------------

def negative_samples(clips: list[Clip], rng: np.random.Generator, n=60):
    """Rough non-sign inputs: idle edges and cross-class splices of real clips."""
    negs = []
    for _ in range(n // 2):  # splice: second half of A + first half of B, different classes
        a, b = rng.choice(len(clips), 2, replace=False)
        while clips[a].label == clips[b].label:
            a, b = rng.choice(len(clips), 2, replace=False)
        da, db = clips[a].df, clips[b].df
        negs.append(("splice", np.concatenate([da.iloc[len(da) // 2:].to_numpy(), db.iloc[: len(db) // 2].to_numpy()]), da.columns))
    for i in rng.choice(len(clips), n // 2, replace=False):  # first 35% of a clip (mostly lead-in)
        d = clips[i].df
        negs.append(("leadin", d.iloc[: max(int(0.35 * len(d)), 20)].to_numpy(), d.columns))
    return negs
