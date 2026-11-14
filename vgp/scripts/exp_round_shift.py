"""Experiment: do pose centring or per-session normalisation fix the cross-round drop?

Usage (from vgp/):  uv run python scripts/exp_round_shift.py
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signrec.data import FeatureConfig as F, load_clips
from signrec.evaluate import grid
clips = load_clips()
cfgs = [F(center="mean"), F(center="rest"), F(handpos=True, center="mean"), F(handpos=True, center="rest"),
        F(handpos=True, trim=True, center="mean"), F(vel=True, center="mean")]
rows = []
r, _ = grid(clips, cfgs); rows += r
r, _ = grid(clips, [F(), F(handpos=True), F(handpos=True, trim=True), F(handpos=True, center="mean")], session_norm=True); rows += r
for r in rows:
    print(f"{r['model']:58s} LORO {r['loro_mean']:.3f} {str(r['loro_per_round']):22s} kfold {r['kfold_mean']:.3f}")
