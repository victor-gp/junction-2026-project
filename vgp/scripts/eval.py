"""Compare DTW k-NN variants with leave-one-round-out and repeated 5-fold CV.

Usage (from vgp/):  uv run python scripts/eval.py
Writes results/summary.json, results/best.json, results/confusion_<split>.png
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signrec.data import FeatureConfig, load_clips, extract  # noqa: E402
from signrec.dtw import DTWKNN  # noqa: E402
from signrec.evaluate import confusion, grid, negative_samples  # noqa: E402
from signrec.plots import plot_confusion, plot_reject  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
CONFIGS = [
    FeatureConfig(),
    FeatureConfig(trim=True),
    FeatureConfig(vel=True),
    FeatureConfig(handpos=True),
    FeatureConfig(handpos=True, vel=True),
    FeatureConfig(handpos=True, trim=True),
]


def main():
    OUT.mkdir(exist_ok=True)
    clips = load_clips()
    classes = sorted({c.label for c in clips})
    print(f"{len(clips)} clips, classes={classes}, rounds={sorted({c.round for c in clips})}")
    rows, details = grid(clips, CONFIGS)

    print(f"\n{'model':44s} {'LORO':>6s} {'per round':>22s} {'5x5-fold':>14s}")
    for r in rows:
        print(f"{r['model']:44s} {r['loro_mean']:6.3f} {str(r['loro_per_round']):>22s} "
              f"{r['kfold_mean']:6.3f}±{r['kfold_std']:.3f}")
    # best: LORO first, then k-fold; list order (simpler first) breaks exact ties
    best = max(rows, key=lambda r: (round(r["loro_mean"], 4), round(r["kfold_mean"], 4)))
    d = details[best["model"]]
    print(f"\nBest: {best['model']}")

    for split in ("loro", "kfold"):
        cm = confusion(clips, d[f"{split}_preds"], classes)
        plot_confusion(cm, classes, OUT / f"confusion_{split}.png",
                       f"{best['model']} - {'leave-one-round-out' if split == 'loro' else '5x5-fold CV'}")
        print(f"confusion ({split}), rows=true, cols=pred {classes}:\n{cm}")

    y = np.array([c.label for c in clips])
    print("\nMisclassified (LORO):")
    correct_d, wrong_d = [], []
    for idx, p, nd in d["loro_preds"]:
        for i, pi, di in zip(idx, p, nd):
            (correct_d if pi == y[i] else wrong_d).append(di)
            if pi != y[i]:
                print(f"  {clips[i].label}/{clips[i].path.name} (round {clips[i].round}) -> {pi}  dist={di:.3f}")

    # reject threshold: 95th percentile of nearest distances for correctly classified LORO clips
    thr = float(np.percentile(correct_d, 95))
    full = DTWKNN(d["cfg"], k=d["k"]).fit([c.df for c in clips], list(y))
    negs = negative_samples(clips, np.random.default_rng(0))
    neg_d = {}
    for kind, arr, cols in negs:
        import pandas as pd
        f = extract(pd.DataFrame(arr, columns=cols), d["cfg"])
        neg_d.setdefault(kind, []).append(float(full.distances_features([f]).min()))
    reject = {"threshold": thr,
              "false_reject_rate_signs": float(np.mean(np.array(correct_d) > thr)),
              "false_accept_rate": {k: float(np.mean(np.array(v) <= thr)) for k, v in neg_d.items()}}
    print(f"\nReject threshold (p95 of correct LORO distances) = {thr:.3f}")
    print(f"  false-accept rate on non-sign inputs: {reject['false_accept_rate']}")
    plot_reject(correct_d, wrong_d, neg_d, thr, OUT / "reject_distances.png")

    summary = {"rows": rows, "best": best["model"], "reject": reject,
               "n_clips": len(clips), "classes": classes}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "best.json").write_text(json.dumps({"cfg": d["cfg"].to_dict(), "k": d["k"],
                                               "reject_threshold": thr}, indent=2))
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
