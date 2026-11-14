"""Live-compatible sign model: 1D-CNN on 3 s windows of the 114 data-capture features.

Fits the data-capture/realtime.py interface:
    model.predict(x)   x: (B, window_frames(), 114) float32 -> (B, n_classes) probabilities
in realtime.classes() order (sorted data/ sub-folders, incl. Idle).

Training needs torch (`uv run --extra cnn`); inference is pure numpy, so the demo
machine only needs numpy + pandas. The saved .npz holds everything needed to rebuild
preprocessing: active-feature mask, normalisation stats, class list, window config.

Also here: a loader that accepts both Studio CSV exports and `main.py record` output,
a NaN-tolerant live feature extractor (freezes a dropped glove), and the vote decoder
shared by the live runner and the offline replay.
"""
from __future__ import annotations

import json
import os
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DIR = REPO_ROOT / "data-capture"
if str(CAPTURE_DIR) not in sys.path:
    sys.path.insert(0, str(CAPTURE_DIR))

import config  # noqa: E402  (data-capture)
import realtime  # noqa: E402
from features import FEATURE_COLUMNS, hand_local_positions, resample  # noqa: E402

N_RAW = len(FEATURE_COLUMNS)                    # 114
HAND_COLS = (np.arange(0, N_RAW // 2), np.arange(N_RAW // 2, N_RAW))   # left, right
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "live_cnn.npz"
# Features with a training std below this (hand-size units) are treated as constant:
# training drops them, and inference pins them to the training mean (see preprocess).
MIN_STD = 1e-2


# -- loading recordings ---------------------------------------------------------

def is_live_recording(df: pd.DataFrame) -> bool:
    """`main.py record` output (SDK quaternions, Timestamp in s) vs Studio export (ms)."""
    if any(c.endswith("_rotation_w") for c in df.columns):
        return True
    ts = df["Timestamp"].to_numpy(dtype=float)
    return len(ts) > 1 and np.median(np.diff(ts)) < 0.5     # Studio: ~33 ms steps


def recording_features(df: pd.DataFrame) -> np.ndarray:
    """(frames @ FEATURE_RATE_HZ, 114) features from either recording format."""
    df = df.sort_values("Timestamp")
    if is_live_recording(df):
        ts, mirror = df["Timestamp"].to_numpy(dtype=float), False
    else:
        ts, mirror = df["Timestamp"].to_numpy(dtype=float) / 1000.0, True
    f = hand_local_positions(df, mirror=mirror)
    f = freeze_missing(f)
    return resample(ts, f)


def file_features(path) -> np.ndarray:
    return recording_features(pd.read_csv(path))


def freeze_missing(f: np.ndarray, last: list | None = None) -> np.ndarray:
    """Per hand, replace NaN rows with the last valid row (glove dropout → frozen hand).
    `last` = [left_row, right_row] carried across calls (updated in place). Leading
    NaNs with no history stay NaN; the model maps them to the training mean."""
    f = f.copy()
    for h, cols in enumerate(HAND_COLS):
        prev = None if last is None else last[h]
        block = f[:, cols]
        bad = np.isnan(block).any(axis=1)
        if not bad.any():
            if last is not None and len(block):
                last[h] = block[-1].copy()
            continue
        for i in range(len(block)):
            if bad[i]:
                if prev is not None:
                    block[i] = prev
            else:
                prev = block[i].copy()
        f[:, cols] = block
        if last is not None:
            last[h] = prev
    return f


class LiveFeatures:
    """Like features.live_features(), but a glove that drops out is frozen at its last
    pose instead of making the whole window NaN."""

    def __init__(self):
        self.last = [None, None]

    def __call__(self, buffer, seconds: float) -> np.ndarray:
        ts, values = buffer.last_seconds(seconds)
        if len(ts) < 2:
            return np.zeros((0, N_RAW), np.float32)
        f = hand_local_positions(pd.DataFrame(values, columns=buffer.columns), mirror=False)
        return resample(ts, freeze_missing(f, self.last))


# -- training windows -------------------------------------------------------------

def dataset_windows(data_dir: str | None = None):
    """All clips in data/ cut into model windows.
    Returns X (N, frames, 114), y (N,), group (N,) clip index, classes, clip paths."""
    data_dir = data_dir or config.DATA_DIR
    classes = realtime.classes()
    xs, ys, gs, paths = [], [], [], []
    for label, name in enumerate(classes):
        folder = os.path.join(data_dir, name)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".csv"))
        if not files:
            print(f"warning: class '{name}' has no CSV clips; it will never be predicted",
                  file=sys.stderr)
        for f in files:
            path = os.path.join(folder, f)
            wins = realtime.clip_windows(file_features(path))
            xs += wins
            ys += [label] * len(wins)
            gs += [len(paths)] * len(wins)
            paths.append(path)
    return (np.stack(xs).astype(np.float32), np.array(ys, np.int64), np.array(gs, np.int64),
            classes, paths)


# -- numpy inference ------------------------------------------------------------------

def _conv1d(x, w, b):                       # x (B, C, T), w (O, C, K), same padding
    k = w.shape[2]
    p = k // 2
    xp = np.pad(x, ((0, 0), (0, 0), (p, p)))
    T = x.shape[2]
    cols = np.stack([xp[:, :, i:i + T] for i in range(k)], axis=-1)    # (B, C, T, K)
    return np.einsum("bctk,ock->bot", cols, w, optimize=True) + b[None, :, None]


def _maxpool2(x):
    T = x.shape[2] // 2 * 2
    return x[:, :, :T].reshape(x.shape[0], x.shape[1], T // 2, 2).max(-1)


@dataclass
class LiveMeta:
    classes: list
    mask: list                 # indices into the 114 features that the model uses
    mean: list
    std: list
    window_frames: int
    feature_rate_hz: float
    width: int
    dropout: float
    trained_on: dict           # class -> number of clips

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class LiveSignModel:
    """predict((B, frames, 114)) -> (B, n_classes) probabilities. Numpy only."""

    # SignCNN layout (signrec.nn): features.{1,2,4,6} are conv blocks (conv, bn), head.1 is linear
    BLOCKS = ("features.1", "features.2", "features.4", "features.6")
    POOL_AFTER = {"features.2", "features.4"}

    def __init__(self, meta: LiveMeta, weights: dict[str, np.ndarray]):
        self.meta = meta
        self.classes = list(meta.classes)
        self.mask = np.asarray(meta.mask, dtype=np.int64)
        self.mean = np.asarray(meta.mean, np.float32)
        self.std = np.asarray(meta.std, np.float32)
        # Near-constant in training (rigid Studio skeleton) but slightly different live:
        # normalised, a tiny live offset becomes a huge value and saturates the model.
        # They carry no information, so pin them to the training mean. Models trained
        # after this fix have none; this protects older ones.
        self.pinned = np.flatnonzero(self.std < MIN_STD)
        self.w = {k: v.astype(np.float32) for k, v in weights.items()}

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)[:, :, self.mask]
        x = (x - self.mean) / self.std
        x[:, :, self.pinned] = 0.0
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)   # missing → training mean

    def logits(self, z: np.ndarray) -> np.ndarray:
        h = z.transpose(0, 2, 1)
        w = self.w
        for name in self.BLOCKS:
            h = _conv1d(h, w[f"{name}.0.weight"], w[f"{name}.0.bias"])
            g, bt = w[f"{name}.1.weight"], w[f"{name}.1.bias"]
            rm, rv = w[f"{name}.1.running_mean"], w[f"{name}.1.running_var"]
            h = (h - rm[None, :, None]) / np.sqrt(rv[None, :, None] + 1e-5) * g[None, :, None] + bt[None, :, None]
            h = np.maximum(h, 0)
            if name in self.POOL_AFTER:
                h = _maxpool2(h)
        h = np.concatenate([h.mean(-1), h.max(-1)], axis=1)
        return h @ w["head.1.weight"].T + w["head.1.bias"]

    def predict(self, x: np.ndarray) -> np.ndarray:
        lg = self.logits(self.preprocess(x))
        e = np.exp(lg - lg.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    # -- persistence -------------------------------------------------------------
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, __meta__=np.array(self.meta.to_json()), **self.w)

    @classmethod
    def load(cls, path=DEFAULT_MODEL, check_classes: bool = True) -> "LiveSignModel":
        z = np.load(path, allow_pickle=False)
        meta = LiveMeta(**json.loads(str(z["__meta__"])))
        m = cls(meta, {k: z[k] for k in z.files if k != "__meta__"})
        if check_classes:
            live = realtime.classes()
            if live != m.classes:
                raise RuntimeError(
                    f"model classes {m.classes} != data/ folders {live}.\n"
                    f"Retrain: cd vgp && uv run --extra cnn python scripts/train_live.py")
            if meta.window_frames != realtime.window_frames():
                raise RuntimeError(f"model window {meta.window_frames} frames != "
                                   f"config window {realtime.window_frames()} frames")
        return m


# -- decoding (shared by live runner and offline replay) -----------------------------------

class VoteDecoder:
    """Same rule as realtime.transcribe: a label must win VOTES consecutive windows
    (top prob >= CONFIDENCE, else Idle); emit it once when it changes."""

    def __init__(self, classes, confidence=None, votes=None, idle=None):
        self.classes = list(classes)
        self.confidence = config.CONFIDENCE if confidence is None else confidence
        self.idle = config.IDLE_LABEL if idle is None else idle
        self.votes: deque[str] = deque(maxlen=config.VOTES if votes is None else votes)
        self.last = self.idle

    def step(self, probs: np.ndarray) -> str | None:
        label = self.classes[int(probs.argmax())] if probs.max() >= self.confidence else self.idle
        self.votes.append(label)
        if len(self.votes) == self.votes.maxlen and self.votes.count(label) == len(self.votes) \
                and label != self.last:
            self.last = label
            if label != self.idle:
                return label
        return None


# -- display ----------------------------------------------------------------------------

GLOSS = {  # data/ folder -> (FinSL gloss, English)
    "Hello": ("MOI", "Hello"),
    "Thank you": ("KIITOS", "Thank you"),
    "No": ("EI", "No"),
    "Okay": ("OKEI", "Okay"),
    "Yes": ("KYLLÄ", "Yes"),
    "Bye": ("HEIHEI", "Bye"),
    "Help": ("AUTTAA", "Help"),
    "World": ("MAAILMA", "World"),
}
PHRASES = {  # consecutive glosses -> English sentence
    ("No", "Thank you"): "No, thank you.",
    ("Okay", "Thank you"): "Okay, thank you.",
    ("Hello", "World"): "Hello, world!",
    ("Thank you", "Help"): "Thanks for the help.",
}


def gloss(label: str) -> str:
    return GLOSS.get(label, (label.upper(), label))[0]


def english(labels: list[str]) -> str:
    """English for a run of words, merging known phrases greedily."""
    out, i = [], 0
    while i < len(labels):
        pair = tuple(labels[i:i + 2])
        if pair in PHRASES:
            out.append(PHRASES[pair])
            i += 2
        else:
            out.append(GLOSS.get(labels[i], (None, labels[i]))[1] + ".")
            i += 1
    return " ".join(out)
