"""Train the final 1D-CNN on all labelled clips and save it to models/cnn_vocab4.pt.

Usage (from vgp/):
    modal run modal_gpu.py --cmd "python scripts/train_cnn.py"
Takes the reject threshold from results/cnn_summary.json if eval_cnn.py has run in the
same environment. On Modal, results/ isn't uploaded, so chain the two:
    modal run modal_gpu.py --cmd "python scripts/eval_cnn.py && python scripts/train_cnn.py"
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signrec.data import load_clips  # noqa: E402
from signrec.nn import CNNClassifier, TrainConfig  # noqa: E402
from eval_cnn import feature_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="angles+handpos")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--reject-threshold", type=float, default=None,
                    help="override; default: results/cnn_summary.json if present")
    ap.add_argument("--out", default=str(ROOT / "models" / "cnn_vocab4.pt"))
    a = ap.parse_args()

    summ = ROOT / "results" / "cnn_summary.json"
    thr = a.reject_threshold
    if thr is None and summ.exists():
        thr = json.loads(summ.read_text())["reject"]["threshold"]
    clips = load_clips()
    m = CNNClassifier(feature_config(a.features), TrainConfig(epochs=a.epochs), reject_threshold=thr)
    m.fit([c.df for c in clips], [c.label for c in clips], verbose=True)
    m.save(a.out)
    print(f"Saved {a.out}  (features={m.cfg.name()}, classes={m.classes}, reject<{thr})")


if __name__ == "__main__":
    main()
