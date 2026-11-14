"""Dynamic Time Warping distance and a DTW k-nearest-neighbour sign classifier."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .data import FeatureConfig, Standardizer, extract


@njit(cache=True)
def dtw_distance(a: np.ndarray, b: np.ndarray, band: float) -> float:
    """DTW with Euclidean frame cost and a Sakoe-Chiba band (fraction of the longer length).

    Returned distance is normalised by (len(a) + len(b)) so clips of different
    lengths are comparable.
    """
    n, m = a.shape[0], b.shape[0]
    w = max(int(band * max(n, m)), abs(n - m))
    inf = np.inf
    D = np.full((n + 1, m + 1), inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - w)
        hi = min(m, i + w)
        for j in range(lo, hi + 1):
            c = 0.0
            for k in range(a.shape[1]):
                d = a[i - 1, k] - b[j - 1, k]
                c += d * d
            c = np.sqrt(c)
            best = D[i - 1, j - 1]
            if D[i - 1, j] < best:
                best = D[i - 1, j]
            if D[i, j - 1] < best:
                best = D[i, j - 1]
            D[i, j] = c + best
    return D[n, m] / (n + m)


def distance_matrix(queries: list[np.ndarray], refs: list[np.ndarray], band: float) -> np.ndarray:
    out = np.empty((len(queries), len(refs)))
    for i, q in enumerate(queries):
        for j, r in enumerate(refs):
            out[i, j] = dtw_distance(q, r, band)
    return out


class DTWKNN:
    """Template-matching classifier. 'Training' = storing standardised templates."""

    def __init__(self, cfg: FeatureConfig | None = None, k: int = 1, band: float = 0.15,
                 reject_threshold: float | None = None):
        self.cfg = cfg or FeatureConfig()
        self.k = k
        self.band = band
        self.reject_threshold = reject_threshold
        self.scaler = Standardizer()
        self.templates: list[np.ndarray] = []
        self.labels: list[str] = []

    # -- training ---------------------------------------------------------
    def fit_features(self, feats: list[np.ndarray], labels: list[str]) -> "DTWKNN":
        self.scaler.fit(feats)
        self.templates = [self.scaler.transform(f) for f in feats]
        self.labels = list(labels)
        return self

    def fit(self, dfs: list[pd.DataFrame], labels: list[str]) -> "DTWKNN":
        return self.fit_features([extract(d, self.cfg) for d in dfs], labels)

    # -- inference --------------------------------------------------------
    def distances_features(self, feats: list[np.ndarray]) -> np.ndarray:
        return distance_matrix([self.scaler.transform(f) for f in feats], self.templates, self.band)

    def decide(self, dists: np.ndarray) -> list[dict]:
        """Turn a (n_queries, n_templates) distance matrix into predictions."""
        labels = np.array(self.labels)
        out = []
        for row in dists:
            order = np.argsort(row)[: self.k]
            votes = Counter(labels[order])
            top = max(votes, key=lambda l: (votes[l], -min(row[order][labels[order] == l])))
            nearest = float(row[order[0]])
            # per-class nearest distance (useful for display / margins)
            per_class = {l: float(row[labels == l].min()) for l in sorted(set(self.labels))}
            rejected = self.reject_threshold is not None and nearest > self.reject_threshold
            out.append({"label": "unknown" if rejected else top, "best_label": top,
                        "nearest_distance": nearest, "per_class_distance": per_class,
                        "rejected": bool(rejected)})
        return out

    def predict(self, df: pd.DataFrame, src_fps: float = 30) -> dict:
        """Classify one clip (DataFrame in training CSV format)."""
        f = extract(df, self.cfg, src_fps=src_fps)
        return self.decide(self.distances_features([f]))[0]

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"type": "DTWKNN", "cfg": self.cfg.to_dict(), "k": self.k, "band": self.band,
                "reject_threshold": self.reject_threshold, "labels": self.labels}
        arrays = {f"t{i}": t for i, t in enumerate(self.templates)}
        np.savez_compressed(path, meta=json.dumps(meta), mean=self.scaler.mean,
                            std=self.scaler.std, **arrays)

    @classmethod
    def load(cls, path: Path) -> "DTWKNN":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        m = cls(FeatureConfig(**meta["cfg"]), meta["k"], meta["band"], meta["reject_threshold"])
        m.scaler.mean, m.scaler.std = z["mean"], z["std"]
        m.labels = meta["labels"]
        m.templates = [z[f"t{i}"] for i in range(len(m.labels))]
        return m
