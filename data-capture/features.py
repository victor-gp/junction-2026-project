"""Common model features for Studio CSV exports (training) and the live buffer (inference).

Both sources use the same hand skeleton but differ in scale, world axes, hand
placement and coordinate-system handedness; rotations are not comparable
(Studio: angles in degrees, unknown convention; SDK: quaternions) and are not used.

Features: every finger bone position, per hand, in a hand-local frame divided by
hand size. That removes scale, axes, heading and hand offset; the mirror is undone
by flipping z of the Studio data. Verified on recorded data: the rigid metacarpal
points match to 1e-3 and normalised bone lengths match exactly.

Hand-local frame (per hand, per frame):
    origin = Hand
    x = Digit2MetaCarpal - Digit5MetaCarpal (across the knuckles), normalised
    z = x cross (Digit3MetaCarpal - Hand), normalised (palm normal)
    y = z cross x (towards the fingers)
    scale = |Digit3ProximalPhalangx - Digit3MetaCarpal| (middle metacarpal bone)

Not represented: wrist angle and hand movement in space (the live solver has neither).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

SIDES = ("Left", "Right")
FINGER_BONES = (
    ["Digit1MetaCarpal", "Digit1ProximalPhalangx", "Digit1DistalPhalanx"]
    + [f"Digit{d}{b}" for d in range(2, 6)
       for b in ("MetaCarpal", "ProximalPhalangx", "IntermediatePhalanx", "DistalPhalanx")]
)
FEATURE_COLUMNS = [f"{s}{b}_{a}" for s in SIDES for b in FINGER_BONES for a in "xyz"]   # 114


def _pos(df: pd.DataFrame, bone: str) -> np.ndarray:
    return df[[f"{bone}_position_{a}" for a in "xyz"]].to_numpy(dtype=np.float64)


def _unit(v: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def hand_local_positions(df: pd.DataFrame, mirror: bool) -> np.ndarray:
    """(frames, 114) features from a DataFrame with <Bone>_position_{x,y,z} columns.
    mirror=True for Studio data."""
    out = []
    for side in SIDES:
        h = _pos(df, f"{side}Hand")
        m2, m3, m5 = (_pos(df, f"{side}Digit{d}MetaCarpal") for d in (2, 3, 5))
        x = _unit(m2 - m5)
        z = _unit(np.cross(x, m3 - h))
        rot = np.stack([x, np.cross(z, x), z], axis=1)       # rows are the local axes
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = 1.0 / np.linalg.norm(_pos(df, f"{side}Digit3ProximalPhalangx") - m3, axis=1)
        for bone in FINGER_BONES:
            v = np.einsum("nij,nj->ni", rot, _pos(df, f"{side}{bone}") - h) * scale[:, None]
            if mirror:
                v[:, 2] *= -1
            out.append(v)
    return np.concatenate(out, axis=1).astype(np.float32)


def resample(ts: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation of x (frames, features) onto a uniform config.FEATURE_RATE_HZ grid."""
    if len(ts) < 2:
        return x
    grid = np.arange(ts[0], ts[-1] + 1e-9, 1.0 / config.FEATURE_RATE_HZ)
    return np.stack([np.interp(grid, ts, x[:, i]) for i in range(x.shape[1])], axis=1).astype(np.float32)


def studio_features(path: str) -> np.ndarray:
    """Features of one Rokoko Studio CSV export (Timestamp in ms)."""
    df = pd.read_csv(path)
    return resample(df["Timestamp"].to_numpy() / 1000.0, hand_local_positions(df, mirror=True))


def live_features(buffer, seconds: float) -> np.ndarray:
    """Features of the last `seconds` of the live FrameBuffer (Timestamp in s)."""
    ts, values = buffer.last_seconds(seconds)
    return resample(ts, hand_local_positions(pd.DataFrame(values, columns=buffer.columns), mirror=False))
