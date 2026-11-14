"""Fit the final DTW k-NN on all labelled clips and save it to models/.

Usage (from vgp/):  uv run python scripts/train.py   (uses results/best.json from eval.py)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signrec.data import FeatureConfig, load_clips  # noqa: E402
from signrec.dtw import DTWKNN  # noqa: E402


def main(out=ROOT / "models" / "dtw_vocab4.npz"):
    best = json.loads((ROOT / "results" / "best.json").read_text())
    clips = load_clips()
    model = DTWKNN(FeatureConfig(**best["cfg"]), k=best["k"], reject_threshold=best["reject_threshold"])
    model.fit([c.df for c in clips], [c.label for c in clips])
    model.save(out)
    print(f"Saved {out} ({len(clips)} templates, cfg={model.cfg.name()}, k={model.k}, "
          f"reject>{model.reject_threshold:.3f})")


if __name__ == "__main__":
    main()
