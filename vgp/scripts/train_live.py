"""Train the live model (CNN on 3 s windows of the 114 live features) -> models/live_cnn.npz.

Training needs torch; the saved model runs with numpy only (see README, "Live demo").

On Modal (from the repo root; models/ and results/ are copied back):
    modal run modal-gpu/run.py --workdir vgp --cmd "python scripts/train_live.py --eval"
Locally on CPU (from vgp/; a few minutes):
    uv run --extra cnn python scripts/train_live.py [--eval] [--smoke]
With a held-out validation set (every 5th clip per class, never trained on):
    modal run modal-gpu/run.py --workdir vgp --cmd "python scripts/train_live.py --eval --holdout 5"
    python scripts/validate_live.py        # from vgp/, scores results/live_holdout.json

Classes are the data/ sub-folders (realtime.classes()), so adding a folder (e.g. Help,
World) just needs a rerun. Accepts Studio CSV exports and `main.py record` output.
CV folds are grouped by clip, so windows of one clip never sit on both sides.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from signrec.live import (DEFAULT_MODEL, HAND_COLS, MIN_STD, LiveMeta,  # noqa: E402
                          LiveSignModel, dataset_windows)
from signrec.nn import SignCNN  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))


@dataclass
class TC:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    width: int = 64
    dropout: float = 0.3
    seed: int = 0
    # augmentation (standardised units)
    noise: float = 0.05
    scale: float = 0.1
    offset: float = 0.3
    warp: float = 0.15          # global speed change +- 15%
    shift: int = 8              # random time shift (frames, edge padded)
    hand_freeze: float = 0.1    # prob. of freezing one hand (glove dropout)


def active_mask(X: np.ndarray, rel: float = 1e-3, min_std: float = MIN_STD) -> np.ndarray:
    """Features that actually vary (drops the rigid metacarpals and finger bases).

    Studio's skeleton is rigid, so some features are constant to ~1e-6 in training but
    differ slightly in the live solver's skeleton. Kept, they get a ~1e-6 std, and a
    1e-3 live offset turns into a z-score in the hundreds, which saturates the model
    (it then always predicted one class at 1.00). Hence the absolute floor, and float64:
    in float32 the rounding error alone is ~4e-4, which let them pass the old threshold."""
    s = X.reshape(-1, X.shape[-1]).astype(np.float64).std(0)
    return np.flatnonzero(s > max(rel * s.max(), min_std))


def augment_batch(x: np.ndarray, tc: TC, rng, hand_groups) -> np.ndarray:
    B, T, D = x.shape
    out = np.empty_like(x)
    grid = np.arange(T)
    for b in range(B):
        speed = 1 + tc.warp * rng.uniform(-1, 1)
        centre = rng.uniform(0, T - 1)
        t = np.clip(centre + (grid - centre) * speed + rng.integers(-tc.shift, tc.shift + 1), 0, T - 1)
        xb = np.stack([np.interp(t, grid, x[b, :, j]) for j in range(D)], axis=1)
        xb = xb * (1 + tc.scale * rng.uniform(-1, 1, D)) + tc.offset * rng.uniform(-1, 1, D)
        xb = xb + tc.noise * rng.standard_normal(xb.shape)
        if rng.random() < tc.hand_freeze:
            g = hand_groups[rng.integers(2)]
            xb[:, g] = xb[rng.integers(T), g]
        out[b] = xb
    return out.astype(np.float32)


def train(X, y, n_classes, tc: TC, verbose=False):
    """X: raw (N, T, 114) windows. Returns (LiveSignModel, torch model on CPU)."""
    rng = np.random.default_rng(tc.seed)
    torch.manual_seed(tc.seed)
    mask = active_mask(X)
    Xm = X[:, :, mask]
    flat = Xm.reshape(-1, Xm.shape[-1]).astype(np.float64)
    mean, std = flat.mean(0).astype(np.float32), np.maximum(flat.std(0), MIN_STD).astype(np.float32)
    Z = ((Xm - mean) / std).astype(np.float32)
    hand_groups = [np.flatnonzero(np.isin(mask, cols)) for cols in HAND_COLS]

    # class-balanced sampling (classes have different numbers of windows)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = 1.0 / np.maximum(counts[y], 1)
    w /= w.sum()

    model = SignCNN(len(mask), n_classes, tc.width, tc.dropout).to(DEVICE)
    steps = int(np.ceil(len(Z) / tc.batch_size))
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc.lr, total_steps=tc.epochs * steps)
    lossf = nn.CrossEntropyLoss(label_smoothing=tc.label_smoothing)
    yt = torch.as_tensor(y, device=DEVICE)
    for ep in range(tc.epochs):
        model.train()
        total = 0.0
        for _ in range(steps):
            idx = rng.choice(len(Z), size=tc.batch_size, p=w)
            xb = torch.as_tensor(augment_batch(Z[idx], tc, rng, hand_groups), device=DEVICE)
            loss = lossf(model(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        if verbose and (ep % 10 == 0 or ep == tc.epochs - 1):
            print(f"    epoch {ep:3d}  loss {total / steps:.4f}", flush=True)
    model = model.cpu().eval()
    weights = {k: v.detach().numpy() for k, v in model.state_dict().items()
               if not k.endswith("num_batches_tracked")}
    meta = LiveMeta(classes=[], mask=mask.tolist(), mean=mean.tolist(), std=std.tolist(),
                    window_frames=X.shape[1], feature_rate_hz=0.0, width=tc.width,
                    dropout=tc.dropout, trained_on={})
    return LiveSignModel(meta, weights), model


HOLDOUT_FILE = "live_holdout.json"


def clip_id(path) -> str:
    """'<Label>/<file>', the same on every machine (local, Modal)."""
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


def holdout_clips(paths, every: int) -> set[int]:
    """Indices of every `every`-th clip per class, in clip-number order. Clip numbers
    follow recording order, so the held-out set spans all recording sessions."""
    def num(p):
        digits = "".join(ch for ch in Path(p).stem if ch.isdigit())
        return int(digits) if digits else 0
    by_class: dict[str, list[int]] = {}
    for i, p in enumerate(paths):
        by_class.setdefault(Path(p).parent.name, []).append(i)
    held = set()
    for idx in by_class.values():
        idx = sorted(idx, key=lambda i: num(paths[i]))
        held.update(idx[every // 2::every])
    return held


def grouped_folds(y, groups, k, seed):
    """Stratified by class, grouped by clip."""
    rng = np.random.default_rng(seed)
    fold_of = {}
    for c in np.unique(y):
        clips = np.unique(groups[y == c])
        rng.shuffle(clips)
        for i, g in enumerate(clips):
            fold_of[g] = i % k
    f = np.array([fold_of[g] for g in groups])
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


def evaluate(X, y, groups, classes, tc: TC, k=5):
    n = len(classes)
    conf_w = np.zeros((n, n), int)
    conf_c = np.zeros((n, n), int)
    for i, (tr, te) in enumerate(grouped_folds(y, groups, k, tc.seed)):
        t0 = time.time()
        m, _ = train(X[tr], y[tr], n, tc)
        p = m.predict(X[te])
        for yy, pp in zip(y[te], p.argmax(1)):
            conf_w[yy, pp] += 1
        for g in np.unique(groups[te]):           # clip-level: mean prob over its windows
            sel = groups[te] == g
            conf_c[y[te][sel][0], p[sel].mean(0).argmax()] += 1
        print(f"  fold {i}: window acc {np.mean(p.argmax(1) == y[te]):.3f}  ({time.time() - t0:.0f}s)",
              flush=True)
    acc_w = np.trace(conf_w) / conf_w.sum()
    acc_c = np.trace(conf_c) / conf_c.sum()
    print(f"\n{k}-fold grouped CV  window acc {acc_w:.3f}   clip acc {acc_c:.3f}")
    print("clip confusion (rows = true):")
    w = max(len(c) for c in classes)
    print(" " * (w + 3) + " ".join(f"{c[:6]:>6}" for c in classes))
    for c, row in zip(classes, conf_c):
        print(f"  {c:<{w}} " + " ".join(f"{v:6d}" for v in row))
    return {"k": k, "window_acc": float(acc_w), "clip_acc": float(acc_c),
            "classes": classes, "clip_confusion": conf_c.tolist(), "window_confusion": conf_w.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=TC.epochs)
    ap.add_argument("--eval", action="store_true", help="grouped k-fold CV before the final fit")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="2 epochs; saves only with --out")
    ap.add_argument("--out", default=None)
    ap.add_argument("--holdout", type=int, default=0, metavar="N",
                    help="keep every Nth clip per class (by clip number) out of CV and the final "
                         "fit, for scripts/validate_live.py; 0 = train on everything")
    a = ap.parse_args()
    tc = TC(epochs=2 if a.smoke else a.epochs)

    t0 = time.time()
    X, y, groups, classes, paths = dataset_windows()
    if a.holdout:
        held = holdout_clips(paths, a.holdout)
        keep = ~np.isin(groups, sorted(held))
        X, y, groups = X[keep], y[keep], groups[keep]
        rel = sorted(clip_id(paths[g]) for g in held)
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / HOLDOUT_FILE).write_text(json.dumps(
            {"every": a.holdout, "classes": classes, "clips": rel}, indent=1))
        print(f"held out {len(rel)} clips (every {a.holdout}th per class) -> results/{HOLDOUT_FILE}")
    per_class = {c: int(len(np.unique(groups[y == i]))) for i, c in enumerate(classes)}
    print(f"{len(X)} windows from {len(np.unique(groups))} clips ({time.time() - t0:.0f}s), device {DEVICE}; "
          f"clips per class {per_class}", flush=True)

    if a.eval:
        res = evaluate(X, y, groups, classes, tc, a.folds)
        res.update(train_config=asdict(tc), clips_per_class=per_class)
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / "live_cv.json").write_text(json.dumps(res, indent=1))

    print("final fit on all windows ...", flush=True)
    m, tm = train(X, y, len(classes), tc, verbose=True)
    import config  # data-capture (on sys.path via signrec.live)
    m.classes = m.meta.classes = classes
    m.meta.trained_on = per_class
    m.meta.feature_rate_hz = config.FEATURE_RATE_HZ

    # the numpy forward pass (used live) must match torch
    xb = X[:16]
    with torch.no_grad():
        pt = torch.softmax(tm(torch.as_tensor(m.preprocess(xb))), -1).numpy()
    diff = float(np.abs(pt - m.predict(xb)).max())
    assert diff < 1e-4, f"numpy/torch mismatch {diff}"
    print(f"train window acc {np.mean(m.predict(X).argmax(1) == y):.3f}; numpy == torch (max diff {diff:.1e})")

    out = a.out or (None if a.smoke else str(DEFAULT_MODEL))
    if out:
        m.save(out)
        print(f"saved {out}  classes={classes}  features={len(m.mask)}/114")


if __name__ == "__main__":
    main()
