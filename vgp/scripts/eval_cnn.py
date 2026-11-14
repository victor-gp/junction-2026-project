"""Cross-validate the 1D-CNN on the same splits as the DTW baseline.

Usage (from vgp/):
    modal run modal_gpu.py --cmd "python scripts/eval_cnn.py"            # on a Modal GPU
    uv run --extra cnn python scripts/eval_cnn.py --smoke                # tiny local check
Options: --features angles+handpos[+vel]  --epochs 200  --seeds 0,1,2  --kfold-repeats 3  --no-aug
Writes results/cnn_summary.json, results/cnn_confusion_{loro,kfold}.png, results/cnn_reject.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signrec.data import FeatureConfig, extract, load_clips  # noqa: E402
from signrec.evaluate import negative_samples, splits_kfold, splits_loro  # noqa: E402
from signrec.nn import CNNClassifier, TrainConfig, default_device  # noqa: E402
from signrec.plots import plot_confusion  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402


def feature_config(spec: str) -> FeatureConfig:
    parts = set(spec.split("+"))
    return FeatureConfig(angles="angles" in parts, handpos="handpos" in parts, vel="vel" in parts,
                         trim="trim" in parts, target_fps=30)


def run_split(name, splits, clips, feats, cfg, tc, seeds):
    """Returns per-group accuracies (mean over seeds) and pooled predictions of the first seed."""
    y = np.array([c.label for c in clips])
    accs, preds = [], []
    for s_i, seed in enumerate(seeds):
        fold_acc = []
        for fold_name, tr, te in splits():
            t0 = time.time()
            m = CNNClassifier(cfg, TrainConfig(**{**tc.to_dict(), "seed": seed}))
            m.fit_features([feats[i] for i in tr], list(y[tr]))
            dec = m.decide(m.proba_features([feats[i] for i in te]))
            p = np.array([d["best_label"] for d in dec])
            conf = np.array([d["confidence"] for d in dec])
            fold_acc.append((fold_name, (p == y[te]).mean(), len(te)))
            if s_i == 0:
                preds.append((te, p, conf))
            print(f"  [{name}] seed {seed} {fold_name:12s} acc {(p == y[te]).mean():.3f}  ({time.time() - t0:.1f}s)", flush=True)
        accs.append(fold_acc)
    return accs, preds


def summarise_loro(accs):
    per_round = np.array([[a for _, a, _ in seed] for seed in accs])   # (seeds, rounds)
    return {"mean": float(per_round.mean()), "per_round": per_round.mean(0).round(3).tolist(),
            "seed_std": float(per_round.mean(1).std())}


def summarise_kfold(accs, n_splits=5):
    reps = []
    for seed in accs:
        for r in range(len(seed) // n_splits):
            chunk = seed[r * n_splits:(r + 1) * n_splits]
            reps.append(sum(a * n for _, a, n in chunk) / sum(n for _, _, n in chunk))
    return {"mean": float(np.mean(reps)), "std": float(np.std(reps))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="angles+handpos")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--kfold-repeats", type=int, default=3)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, 1 seed, LORO only (code check)")
    ap.add_argument("--tag", default="", help="suffix for output files")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.seeds = 2, "0"

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ""
    cfg = feature_config(a.features)
    tc = TrainConfig(epochs=a.epochs, aug=not a.no_aug)
    seeds = [int(s) for s in a.seeds.split(",")]
    clips = load_clips()
    classes = sorted({c.label for c in clips})
    y = np.array([c.label for c in clips])
    feats = [extract(c.df, cfg) for c in clips]
    print(f"device={default_device()}  clips={len(clips)}  features={cfg.name()} dim={feats[0].shape[1]}  "
          f"epochs={tc.epochs} aug={tc.aug} seeds={seeds}", flush=True)

    t0 = time.time()
    loro_accs, loro_preds = run_split("loro", lambda: splits_loro(clips), clips, feats, cfg, tc, seeds)
    loro = summarise_loro(loro_accs)
    kfold, kf_preds = None, None
    if not a.smoke:
        kf_accs, kf_preds = run_split("kfold", lambda: splits_kfold(clips, n_repeats=a.kfold_repeats),
                                      clips, feats, cfg, tc, seeds)
        kfold = summarise_kfold(kf_accs)

    def pooled_cm(preds):
        return confusion_matrix(np.concatenate([y[i] for i, _, _ in preds]),
                                np.concatenate([p for _, p, _ in preds]), labels=classes)

    name = f"cnn {cfg.name()}{'' if tc.aug else ' no-aug'}"
    plot_confusion(pooled_cm(loro_preds), classes, out / f"cnn_confusion_loro{tag}.png",
                   f"{name} - leave-one-round-out")
    if kf_preds:
        plot_confusion(pooled_cm(kf_preds), classes, out / f"cnn_confusion_kfold{tag}.png", f"{name} - k-fold CV")

    # reject threshold: 5th percentile of confidence on correctly classified LORO clips
    correct_conf = [c for i, p, cf in loro_preds for yi, pi, c in zip(y[i], p, cf) if yi == pi]
    thr = float(np.percentile(correct_conf, 5)) if correct_conf else None
    reject = {"threshold": thr}
    if not a.smoke and thr is not None:
        full = CNNClassifier(cfg, tc).fit_features(feats, list(y))
        fa = {}
        for kind, arr, cols in negative_samples(clips, np.random.default_rng(0)):
            conf = full.proba_features([extract(pd.DataFrame(arr, columns=cols), cfg)]).max()
            fa.setdefault(kind, []).append(conf >= thr)
        reject["false_accept_rate"] = {k: float(np.mean(v)) for k, v in fa.items()}

    summary = {"model": name, "features": cfg.to_dict(), "train": tc.to_dict(), "seeds": seeds,
               "loro": loro, "kfold": kfold, "reject": reject, "classes": classes,
               "confusion_loro": pooled_cm(loro_preds).tolist(),
               "device": str(default_device()), "seconds": round(time.time() - t0, 1)}
    (out / f"cnn_summary{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{name}\n  LORO  {loro['mean']:.3f}  per round {loro['per_round']}  (seed std {loro['seed_std']:.3f})")
    if kfold:
        print(f"  kfold {kfold['mean']:.3f} ± {kfold['std']:.3f}")
    print(f"  reject: {reject}\n  confusion (LORO, rows=true {classes}):\n{pooled_cm(loro_preds)}")
    print(f"  total {summary['seconds']}s -> {out}/cnn_summary{tag}.json")


if __name__ == "__main__":
    main()
