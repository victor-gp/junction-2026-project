"""Record labelled training clips through the live glove path, with on-screen prompts.

The model was trained on Rokoko Studio exports, but the demo runs on the live SDK
solver, which places the fingers slightly differently. Clips recorded here are in
the live format (the loader detects it), so training on them closes that gap.

From vgp/, gloves connected, Rokoko Studio closed:
    python scripts/record_prompted.py                      # 5 takes of every class
    python scripts/record_prompted.py --reps 10 --classes Hello No
    python scripts/record_prompted.py --dry-run            # show the plan only

For each take: "get ready" countdown, then RECORD for --seconds. Sign once and
return to rest. For Idle, just rest or fidget naturally. Each take is saved to
data/<Label>/live_<session>_<n>.csv. Then retrain (see README, "Retraining").
"""
import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from signrec.live import config, gloss  # noqa: E402  (also puts data-capture on sys.path)


def countdown(label: str, seconds: float) -> None:
    for s in range(int(seconds), 0, -1):
        print(f"\r  next: {label:<10} ({gloss(label)}) in {s} ...   ", end="", flush=True)
        time.sleep(1)
    print(f"\r  >>> {label.upper():<10} NOW ({gloss(label)})            ", end="", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", nargs="+", help="default: all data/ folders")
    ap.add_argument("--reps", type=int, default=5, help="takes per class (default 5)")
    ap.add_argument("--seconds", type=float, default=4.0, help="length of one take (default 4)")
    ap.add_argument("--ready", type=float, default=3.0, help="countdown before each take (default 3)")
    ap.add_argument("--data-dir", default=config.DATA_DIR)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import realtime  # data-capture
    classes = a.classes or realtime.classes()
    plan = [c for c in classes for _ in range(a.reps)]
    random.shuffle(plan)       # mixed order, so takes don't all share one moment's posture
    session = time.strftime("%Y%m%d_%H%M%S")
    print(f"{len(plan)} takes of {a.seconds:.0f} s: {', '.join(classes)} x {a.reps}, "
          f"saved as {a.data_dir}/<Label>/live_{session}_<n>.csv")
    if a.dry_run:
        print("order:", " ".join(plan))
        return

    from rokoko_stream import GloveStreamReader   # Windows + Rokoko Device SDK
    with GloveStreamReader() as reader:
        print("waiting for both gloves ...", flush=True)
        t_end = time.time() + 30
        while len(reader.connected_hands) < 2 and time.time() < t_end:
            time.sleep(0.5)
        if len(reader.connected_hands) < 2:
            sys.exit(f"only {reader.connected_hands or 'no'} gloves connected"
                     + (f" ({reader.last_error})" if reader.last_error else ""))
        print("both gloves connected. Ctrl+C stops; finished takes are kept.\n")
        done = 0
        try:
            for i, label in enumerate(plan, 1):
                countdown(label, a.ready)
                start = time.time()
                time.sleep(a.seconds)
                ts, _ = reader.buffer.latest()
                df = reader.buffer.to_dataframe()
                df = df[df["Timestamp"] >= ts[-1] - (time.time() - start)]
                out = Path(a.data_dir) / label / f"live_{session}_{i:03d}.csv"
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out, index=False)
                done += 1
                nan = df.drop(columns="Timestamp").isna().all(axis=1).mean()
                print(f"\r  [{i}/{len(plan)}] {label:<10} {len(df)} frames -> {out.name}"
                      + (f"   WARNING: {nan:.0%} frames without hands" if nan > 0.05 else "") + " " * 10)
        except KeyboardInterrupt:
            print("\nstopped.")
        print(f"\n{done} takes saved. Retrain: see README, 'Retraining'.")


if __name__ == "__main__":
    main()
