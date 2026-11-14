"""Entry point. Run from anywhere:

    python data-capture/main.py windows --out windows.npz       # training data from data/ (Studio CSVs)
    python data-capture/main.py live --model my_model:load      # real-time transcription
    python data-capture/main.py record --seconds 5 --out rec.csv  # dump the live buffer to CSV

--model "module:attr": attr is the model object or a zero-argument function returning it
(module is imported from the current directory). It needs predict(x) -> probabilities,
see realtime.py. Parameters live in config.py.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import config  # noqa: E402
import realtime  # noqa: E402
from features import FEATURE_COLUMNS  # noqa: E402
from rokoko_stream import GloveStreamReader  # noqa: E402


def load_model(spec: str) -> realtime.Model:
    sys.path.insert(0, os.getcwd())
    module, _, attr = spec.partition(":")
    obj = getattr(importlib.import_module(module), attr or "model")
    return obj if hasattr(obj, "predict") else obj()


def cmd_windows(args) -> None:
    X, y = realtime.training_windows()
    np.savez_compressed(args.out, X=X, y=y, classes=realtime.classes(), feature_columns=FEATURE_COLUMNS)
    counts = dict(zip(realtime.classes(), np.bincount(y).tolist()))
    print(f"{X.shape[0]} windows of {X.shape[1]}x{X.shape[2]} -> {args.out}  {counts}")


def cmd_live(args) -> None:
    model = load_model(args.model)
    with GloveStreamReader() as reader:
        print(f"Listening for {', '.join(realtime.classes())} ... Ctrl+C to stop")
        try:
            realtime.transcribe(model, reader.buffer)
        except KeyboardInterrupt:
            pass


def cmd_record(args) -> None:
    with GloveStreamReader() as reader:
        time.sleep(args.seconds)
        df = reader.buffer.to_dataframe()
        df.to_csv(args.out, index=False)
        print(f"{len(df)} frames, hands {reader.connected_hands} -> {args.out}"
              + (f"  (error: {reader.last_error})" if reader.last_error else ""))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("windows", help="cut Studio clips into model windows (.npz)")
    w.add_argument("--out", default="windows.npz")
    lv = sub.add_parser("live", help="real-time transcription")
    lv.add_argument("--model", required=True, help='"module:attr", see above')
    r = sub.add_parser("record", help="record the live buffer to CSV")
    r.add_argument("--seconds", type=float, default=5.0)
    r.add_argument("--out", default="recording.csv")
    args = p.parse_args()
    {"windows": cmd_windows, "live": cmd_live, "record": cmd_record}[args.cmd](args)


if __name__ == "__main__":
    main()
