"""Score the live model on the clips that train_live.py --holdout kept out of training.

From vgp/ (numpy + pandas only):
    python scripts/validate_live.py                       # clips from results/live_holdout.json
    python scripts/validate_live.py --clips "../data/No/Clip_150.csv" ...

Per clip, three views:
- window: fraction of 3 s windows whose top class is right,
- clip:   class with the highest mean probability over the clip's windows,
- live:   words the live vote decoder (same rule as scripts/live.py) emits when the
          clip is replayed; "ok" = only the right word fired (Idle clips: nothing fired).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from signrec.live import (DEFAULT_MODEL, LiveSignModel, VoteDecoder, config,  # noqa: E402
                          file_features, realtime)

HOLDOUT = ROOT / "results" / "live_holdout.json"


def live_words(model, feats, decoder_kw) -> list[str]:
    """Words the live decoder emits on this clip (same windows as scripts/live.py --replay)."""
    rate, n = config.FEATURE_RATE_HZ, realtime.window_frames()
    ends = np.arange(config.WINDOW_S * config.MIN_FILL, len(feats) / rate + 1e-9, config.HOP_S)
    if not len(ends):
        return []
    idx = np.round(ends * rate).astype(int)
    probs = model.predict(np.stack([realtime.fit_window(feats[max(0, e - n):e]) for e in idx]))
    dec = VoteDecoder(model.classes, **decoder_kw)
    return [w for w in (dec.step(p) for p in probs) if w]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--clips", nargs="+", help="clip CSVs (default: results/live_holdout.json)")
    ap.add_argument("--confidence", type=float, default=None)
    ap.add_argument("--votes", type=int, default=None)
    a = ap.parse_args()

    model = LiveSignModel.load(a.model, check_classes=False)
    if a.clips:
        clips = [Path(c) for c in a.clips]
    else:
        clips = [Path(config.DATA_DIR) / c for c in json.loads(HOLDOUT.read_text())["clips"]]
    clips = [c for c in clips if c.parent.name in model.classes]
    trained = model.meta.trained_on
    print(f"model {a.model}: {model.classes}, trained on {sum(trained.values())} clips {trained}")
    print(f"validating on {len(clips)} held-out clips\n")

    n, kw = len(model.classes), {"confidence": a.confidence, "votes": a.votes}
    conf = np.zeros((n, n), int)
    win_ok = win_tot = live_ok = 0
    w = max(len(c) for c in model.classes)
    for c in clips:
        truth = c.parent.name
        feats = file_features(c)
        p = model.predict(np.stack(realtime.clip_windows(feats)))
        t = model.classes.index(truth)
        pred = int(p.mean(0).argmax())
        conf[t, pred] += 1
        win_ok += int((p.argmax(1) == t).sum())
        win_tot += len(p)
        words = live_words(model, feats, kw)
        ok = words == ([] if truth == config.IDLE_LABEL else [truth])
        live_ok += ok
        print(f"  {truth + '/' + c.name:<{w + 14}} clip {'✓' if pred == t else '✗'} {model.classes[pred]:<{w}}"
              f"  windows {np.mean(p.argmax(1) == t):4.0%}   live {'ok ' if ok else 'BAD'} {words or '-'}")

    print(f"\nclip acc {np.trace(conf)}/{conf.sum()} = {np.trace(conf) / conf.sum():.1%}   "
          f"window acc {win_ok / win_tot:.1%}   live ok {live_ok}/{len(clips)} = {live_ok / len(clips):.1%}")
    print("clip confusion (rows = true):")
    print(" " * (w + 3) + " ".join(f"{c[:6]:>6}" for c in model.classes))
    for c, row in zip(model.classes, conf):
        print(f"  {c:<{w}} " + " ".join(f"{v:6d}" for v in row))


if __name__ == "__main__":
    main()
