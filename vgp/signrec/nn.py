"""Small 1D-CNN sign classifier with on-the-fly augmentation (PyTorch).

Same input contract and predict() output shape as signrec.dtw.DTWKNN, so the
two can be swapped in a live pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import FeatureConfig, Standardizer, extract, resample


@dataclass
class TrainConfig:
    seq_len: int = 64           # every clip is resampled to this many frames
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    width: int = 64             # channels of the first conv block
    dropout: float = 0.3
    seed: int = 0
    # augmentation (applied to standardised features, training only)
    aug: bool = True
    crop_min: float = 0.8       # random temporal crop keeps 80-100% of the clip
    warp: float = 0.2           # strength of smooth random time warping
    noise: float = 0.05         # gaussian noise (std, in z-units)
    scale: float = 0.1          # per-channel gain 1 +- scale
    offset: float = 0.3         # per-channel offset +- offset (mimics calibration / signer shifts)
    hand_dropout: float = 0.1   # prob. of freezing one hand's channels (glove stops emitting)

    def to_dict(self):
        return asdict(self)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- model ----------------------------------------------------------------

class SignCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int, width: int = 64, dropout: float = 0.3):
        super().__init__()

        def block(cin, cout, k):
            return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout),
                                 nn.ReLU(), nn.Dropout(dropout / 2))

        self.features = nn.Sequential(
            nn.Dropout(0.1),                      # light input dropout over channels/time
            block(n_features, width, 5),
            block(width, width, 5), nn.MaxPool1d(2),
            block(width, 2 * width, 3), nn.MaxPool1d(2),
            block(2 * width, 2 * width, 3),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * width, n_classes))

    def forward(self, x):                         # x: (B, T, D)
        h = self.features(x.transpose(1, 2))      # (B, C, T')
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(h)


# -- augmentation -----------------------------------------------------------

def _hand_channel_groups(cfg: FeatureConfig, n_features: int):
    """Indices of left- and right-hand channels, for hand dropout. None if unknown."""
    if not cfg.angles:
        return None
    # angle column order: L wrist(3), R wrist(3), L digits 1-5 (57), R digits 1-5 (57)
    left = list(range(0, 3)) + list(range(6, 63))
    right = list(range(3, 6)) + list(range(63, 120))
    if cfg.handpos:                               # +9: L rel (3), R rel (3), R-L (3)
        left += [120, 121, 122]
        right += [123, 124, 125]
    if cfg.vel:
        base = n_features // 2
        left += [i + base for i in left]
        right += [i + base for i in right]
    return np.array(left), np.array(right)


def augment(x: np.ndarray, tc: TrainConfig, rng: np.random.Generator, groups) -> np.ndarray:
    """x: (T, D) standardised features -> (seq_len, D) augmented."""
    T = len(x)
    # temporal crop
    keep = int(round(T * rng.uniform(tc.crop_min, 1.0)))
    start = rng.integers(0, T - keep + 1)
    x = x[start:start + keep]
    # smooth monotonic time warp: cumulative sum of positive random speeds
    knots = 1.0 + tc.warp * rng.uniform(-1, 1, size=4)
    speed = np.interp(np.linspace(0, 3, tc.seq_len), np.arange(4), knots)
    t = np.cumsum(speed)
    t = (t - t[0]) / (t[-1] - t[0]) * (len(x) - 1)
    x = np.stack([np.interp(t, np.arange(len(x)), x[:, j]) for j in range(x.shape[1])], axis=1)
    D = x.shape[1]
    x = x * (1 + tc.scale * rng.uniform(-1, 1, D)) + tc.offset * rng.uniform(-1, 1, D)
    x = x + tc.noise * rng.standard_normal(x.shape)
    if groups is not None and rng.random() < tc.hand_dropout:
        g = groups[rng.integers(2)]
        x[:, g] = x[0, g]                         # freeze: glove stopped emitting
    return x.astype(np.float32)


def to_fixed(x: np.ndarray, seq_len: int) -> np.ndarray:
    return resample(x, len(x), seq_len).astype(np.float32)


# -- classifier ------------------------------------------------------------

class CNNClassifier:
    def __init__(self, cfg: FeatureConfig | None = None, tc: TrainConfig | None = None,
                 reject_threshold: float | None = None, device: torch.device | None = None):
        # 30 fps features; the fixed-length resample happens later
        self.cfg = cfg or FeatureConfig(handpos=True, target_fps=30)
        self.tc = tc or TrainConfig()
        self.reject_threshold = reject_threshold   # min softmax confidence to accept
        self.device = device or default_device()
        self.scaler = Standardizer()
        self.classes: list[str] = []
        self.model: SignCNN | None = None

    # -- training ---------------------------------------------------------
    def fit_features(self, feats: list[np.ndarray], labels: list[str], verbose: bool = False) -> "CNNClassifier":
        tc = self.tc
        rng = np.random.default_rng(tc.seed)
        torch.manual_seed(tc.seed)
        self.classes = sorted(set(labels))
        y = np.array([self.classes.index(l) for l in labels])
        self.scaler.fit(feats)
        xs = [self.scaler.transform(f) for f in feats]
        D = xs[0].shape[1]
        groups = _hand_channel_groups(self.cfg, D)
        fixed = np.stack([to_fixed(x, tc.seq_len) for x in xs])   # used when aug is off

        self.model = SignCNN(D, len(self.classes), tc.width, tc.dropout).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=tc.lr, total_steps=tc.epochs * int(np.ceil(len(xs) / tc.batch_size)))
        lossf = nn.CrossEntropyLoss(label_smoothing=tc.label_smoothing)
        yt = torch.as_tensor(y, device=self.device)

        for ep in range(tc.epochs):
            self.model.train()
            order = rng.permutation(len(xs))
            total = 0.0
            for b in range(0, len(order), tc.batch_size):
                idx = order[b:b + tc.batch_size]
                if len(idx) < 2:                  # BatchNorm needs >1 sample
                    continue
                if tc.aug:
                    xb = np.stack([augment(xs[i], tc, rng, groups) for i in idx])
                else:
                    xb = fixed[idx]
                xb = torch.as_tensor(xb, device=self.device)
                loss = lossf(self.model(xb), yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
            if verbose and (ep % 25 == 0 or ep == tc.epochs - 1):
                print(f"    epoch {ep:4d}  loss {total / len(xs):.4f}", flush=True)
        return self

    def fit(self, dfs: list[pd.DataFrame], labels: list[str], verbose: bool = False) -> "CNNClassifier":
        return self.fit_features([extract(d, self.cfg) for d in dfs], labels, verbose)

    # -- inference --------------------------------------------------------
    @torch.no_grad()
    def proba_features(self, feats: list[np.ndarray]) -> np.ndarray:
        self.model.eval()
        x = np.stack([to_fixed(self.scaler.transform(f), self.tc.seq_len) for f in feats])
        logits = self.model(torch.as_tensor(x, device=self.device))
        return torch.softmax(logits, -1).cpu().numpy()

    def decide(self, probs: np.ndarray) -> list[dict]:
        out = []
        for p in probs:
            top = self.classes[int(p.argmax())]
            conf = float(p.max())
            rejected = self.reject_threshold is not None and conf < self.reject_threshold
            out.append({"label": "unknown" if rejected else top, "best_label": top,
                        "confidence": conf, "probs": dict(zip(self.classes, map(float, p))),
                        "rejected": bool(rejected)})
        return out

    def predict(self, df: pd.DataFrame, src_fps: float = 30) -> dict:
        return self.decide(self.proba_features([extract(df, self.cfg, src_fps=src_fps)]))[0]

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"type": "CNNClassifier", "cfg": self.cfg.to_dict(), "tc": self.tc.to_dict(),
                    "classes": self.classes, "reject_threshold": self.reject_threshold,
                    "mean": self.scaler.mean, "std": self.scaler.std,
                    "n_features": int(self.scaler.mean.shape[0]),
                    "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()}}, path)

    @classmethod
    def load(cls, path: Path, device: torch.device | None = None) -> "CNNClassifier":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(FeatureConfig(**ck["cfg"]), TrainConfig(**ck["tc"]), ck["reject_threshold"], device)
        m.classes = ck["classes"]
        m.scaler.mean, m.scaler.std = ck["mean"], ck["std"]
        m.model = SignCNN(ck["n_features"], len(m.classes), m.tc.width, m.tc.dropout)
        m.model.load_state_dict(ck["state_dict"])
        m.model.to(m.device).eval()
        return m
