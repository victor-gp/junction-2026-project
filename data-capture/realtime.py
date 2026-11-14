"""Model windows (training and live) and the real-time transcription loop.

Training and live input have the same shape: (window_frames(), 114).

The model is a black box with one method:
    model.predict(x)  x: (batch, window_frames(), 114) float32
                      -> (batch, len(classes())) class probabilities
classes() are the sub-folder names of config.DATA_DIR, sorted; the model's
output order must match.
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Callable, Protocol

import numpy as np

import config
from features import live_features, studio_features


class Model(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...


def classes() -> list[str]:
    d = config.DATA_DIR
    return sorted(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n)))


def window_frames() -> int:
    return int(round(config.WINDOW_S * config.FEATURE_RATE_HZ))


def fit_window(x: np.ndarray) -> np.ndarray:
    """Exactly window_frames() frames: keep the newest, or pad the front with the first frame."""
    n = window_frames()
    return x[-n:] if len(x) >= n else np.pad(x, ((n - len(x), 0), (0, 0)), mode="edge")


def clip_windows(x: np.ndarray) -> list[np.ndarray]:
    """Cut one clip's features into overlapping model windows."""
    n, rate = window_frames(), config.FEATURE_RATE_HZ
    trim = int(round(config.EDGE_TRIM_S * rate))
    if trim and len(x) > 2 * trim:
        x = x[trim:-trim]
    hop = max(1, int(round(config.TRAIN_HOP_S * rate)))
    return [x[i:i + n] for i in range(0, len(x) - n + 1, hop)] or [fit_window(x)]


def training_windows() -> tuple[np.ndarray, np.ndarray]:
    """All Studio clips cut into windows: X (windows, frames, 114) float32, y (class index)."""
    xs, ys = [], []
    for label, name in enumerate(classes()):
        folder = os.path.join(config.DATA_DIR, name)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(".csv"):
                wins = clip_windows(studio_features(os.path.join(folder, f)))
                xs += wins
                ys += [label] * len(wins)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def transcribe(model: Model, buffer, on_word: Callable[[str], None] = print,
               should_stop: Callable[[], bool] = lambda: False) -> None:
    """Run the model on sliding windows of `buffer`; call on_word for each detected sign."""
    labels = classes()
    votes: deque[str] = deque(maxlen=config.VOTES)
    last = config.IDLE_LABEL
    while not should_stop():
        time.sleep(config.HOP_S)
        x = live_features(buffer, config.WINDOW_S)
        if len(x) < config.MIN_FILL * window_frames() or np.isnan(x).any():
            continue                                  # not enough data yet, or a glove missing
        probs = model.predict(fit_window(x)[None])[0]
        label = labels[int(probs.argmax())] if probs.max() >= config.CONFIDENCE else config.IDLE_LABEL
        votes.append(label)
        if len(votes) == votes.maxlen and votes.count(label) == len(votes) and label != last:
            last = label
            if label != config.IDLE_LABEL:
                on_word(label)
