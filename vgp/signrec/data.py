"""Clip loading and feature extraction.

Input contract (training CSVs and live input alike): a DataFrame in the Rokoko
Studio-style export format, 30 fps, with the 241 columns of data/<Label>/Clip_N.csv.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SOURCE_FPS = 30
EXCLUDE_DIRS = {"Test"}

# Recording rounds, inferred from the global clip numbering (see notes):
# round 1 = clips 7-29, round 2 = 31-53, round 3 = 55-83.
ROUND_EDGES = (30, 54)


@dataclass
class Clip:
    label: str
    path: Path
    clip_no: int
    round: int
    df: pd.DataFrame


def round_of(clip_no: int) -> int:
    return 1 + sum(clip_no > e for e in ROUND_EDGES)


def load_clips(data_dir: Path = DATA_DIR) -> list[Clip]:
    clips = []
    for path in sorted(data_dir.glob("*/*.csv")):
        label = path.parent.name
        if label in EXCLUDE_DIRS:
            continue
        m = re.search(r"Clip_(\d+)", path.stem)
        clip_no = int(m.group(1)) if m else 0
        clips.append(Clip(label, path, clip_no, round_of(clip_no), pd.read_csv(path)))
    return clips


def angle_columns(columns) -> list[str]:
    return [c for c in columns if c != "Timestamp" and "_position_" not in c]


HAND_POS = [f"{side}Hand_position_{a}" for side in ("Left", "Right") for a in "xyz"]


@dataclass
class FeatureConfig:
    angles: bool = True        # 120 joint angles (degrees)
    handpos: bool = False      # wrist positions rel. to frame 0 + left->right vector (9 dims)
    vel: bool = False          # append first differences of the above
    trim: bool = False         # trim idle edges by motion energy
    center: str = "none"       # "none" | "mean" (subtract clip mean pose) | "rest" (subtract first-frames pose)
    target_fps: int = 15       # resample (30 fps source); lower = faster DTW

    def name(self) -> str:
        parts = [p for p in ("angles", "handpos", "vel", "trim") if getattr(self, p)]
        if self.center != "none":
            parts.append(f"center-{self.center}")
        return "+".join(parts) + f"@{self.target_fps}fps"

    def to_dict(self):
        return asdict(self)


def idle_bounds(x: np.ndarray, rel_thresh: float = 0.1, margin: int = 3, min_keep: float = 0.6) -> tuple[int, int]:
    """(start, end) frame window without low-motion edges. Never keeps less than `min_keep` of the clip."""
    n = len(x)
    if n < 10:
        return 0, n
    e = np.abs(np.diff(x, axis=0)).sum(1)
    e = np.convolve(e, np.ones(5) / 5, mode="same")
    active = np.where(e > rel_thresh * e.max())[0]
    if len(active) == 0:
        return 0, n
    a, b = max(active[0] - margin, 0), min(active[-1] + margin + 2, n)
    return (a, b) if b - a >= min_keep * n else (0, n)


def resample(x: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if src_fps == dst_fps:
        return x
    n_out = max(int(round(len(x) * dst_fps / src_fps)), 2)
    t_in = np.linspace(0, 1, len(x))
    t_out = np.linspace(0, 1, n_out)
    return np.stack([np.interp(t_out, t_in, x[:, j]) for j in range(x.shape[1])], axis=1)


def extract(df: pd.DataFrame, cfg: FeatureConfig, src_fps: float = SOURCE_FPS) -> np.ndarray:
    """DataFrame (training CSV format) -> (T, D) float32 feature array, unnormalised."""
    blocks = []
    if cfg.angles:
        blocks.append(df[angle_columns(df.columns)].to_numpy(np.float64))
    if cfg.handpos:
        p = df[HAND_POS].to_numpy(np.float64)
        left, right = p[:, :3], p[:, 3:]
        blocks += [left - left[0], right - right[0], right - left]
    x = np.concatenate(blocks, axis=1)
    if cfg.center == "mean":
        x = x - x.mean(0)
    elif cfg.center == "rest":
        x = x - x[:5].mean(0)
    if cfg.trim:
        # detect idle edges on angles only (positions drift), then cut all features
        a, b = idle_bounds(x[:, :120] if cfg.angles else x)
        x = x[a:b]
    x = resample(x, src_fps, cfg.target_fps)
    if cfg.vel:
        x = np.concatenate([x, np.vstack([np.zeros((1, x.shape[1])), np.diff(x, axis=0)])], axis=1)
    return x.astype(np.float32)


@dataclass
class Standardizer:
    mean: np.ndarray = field(default=None)
    std: np.ndarray = field(default=None)

    def fit(self, seqs: list[np.ndarray]) -> "Standardizer":
        allf = np.concatenate(seqs, axis=0)
        self.mean = allf.mean(0)
        self.std = np.maximum(allf.std(0), 1e-3 * (allf.std(0).max() + 1e-9))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)
