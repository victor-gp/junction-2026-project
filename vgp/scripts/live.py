"""Live sign transcription from the gloves, or offline replay of a recording.

From vgp/ (numpy + pandas only; no torch needed):
    python scripts/live.py                          # live from the gloves (Windows + Rokoko Device SDK)
    python scripts/live.py --replay take.csv [...]  # offline: Studio CSV or `main.py record` output
    python scripts/live.py --replay take.csv --trace   # + top class per window

Same windowing and vote rule as data-capture/realtime.transcribe, with two changes:
a glove that drops out is frozen at its last pose (instead of stopping recognition), and
words are shown as FinSL gloss + English. Knobs: --confidence, --votes, --hop
(defaults from data-capture/config.py).
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from signrec.live import (DEFAULT_MODEL, Adapter, LiveFeatures, LiveSignModel, VoteDecoder,  # noqa: E402
                          config, english, file_features, gloss, realtime)

PAUSE_S = 5.0    # no word for this long → start a new utterance line


class Transcript:
    def __init__(self):
        self.words: list[str] = []
        self.last_t = -1e9

    def add(self, label: str, t: float) -> str:
        if t - self.last_t > PAUSE_S:
            self.words = []
        self.words.append(label)
        self.last_t = t
        glosses = " · ".join(gloss(w) for w in self.words)
        return f"{glosses:<30} →  \"{english(self.words)}\""


def show(t: float, label: str, transcript: Transcript) -> None:
    print(f"[{t:6.1f}s] {gloss(label):<8} | {transcript.add(label, t)}", flush=True)


def replay(model, path: str, decoder_kw: dict, trace: bool, adapt: float = 0.0) -> list[tuple[float, str]]:
    """Slide the live window over a whole recording, exactly as live would see it."""
    feats = file_features(path)
    rate, n = config.FEATURE_RATE_HZ, realtime.window_frames()
    ends = np.arange(config.WINDOW_S * config.MIN_FILL, len(feats) / rate + 1e-9, config.HOP_S)
    ends_idx = np.round(ends * rate).astype(int)
    X = np.stack([realtime.fit_window(feats[max(0, e - n):e]) for e in ends_idx])
    if adapt:   # causal, like live: each window only sees the frames up to its end
        ad, prev = Adapter(model, adapt), 0
        for i, e in enumerate(ends_idx):
            ad.push(feats[prev:e])
            prev = e
            X[i] = ad(X[i])
    probs = model.predict(X)
    dec, tr, words = VoteDecoder(model.classes, **decoder_kw), Transcript(), []
    print(f"\n== {path}  ({len(feats) / rate:.1f}s)")
    for t, p in zip(ends, probs):
        if trace:
            top = np.argsort(p)[::-1][:2]
            print(f"   {t:6.2f}s  " + "  ".join(f"{model.classes[i]:<10}{p[i]:.2f}" for i in top))
        w = dec.step(p)
        if w:
            show(t, w, tr)
            words.append((t, w))
    if not words:
        print("   (no words)")
    return words


class SessionLog:
    """<dir>/<stamp>_predictions.csv: one row per model window (time, all class
    probabilities, top class, emitted word). <dir>/<stamp>_raw.csv: every solver frame,
    in the `main.py record` format, so the session can be replayed with --replay.
    Both are flushed every hop, so a killed run keeps everything up to that point."""

    def __init__(self, log_dir: str, classes, columns):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.pred_path = Path(log_dir) / f"{stamp}_predictions.csv"
        self.raw_path = Path(log_dir) / f"{stamp}_raw.csv"
        self.classes = list(classes)
        self.pred = open(self.pred_path, "w", encoding="utf-8", newline="")
        self.pred.write("t_s,stream_ts,hands,frames," + ",".join(f"p_{c}" for c in self.classes)
                        + ",top,top_p,word\n")
        self.raw = open(self.raw_path, "w", encoding="utf-8", newline="")
        self.raw.write("Timestamp," + ",".join(columns) + "\n")
        self.last_ts = -np.inf
        print(f"logging to {self.pred_path} and {self.raw_path}", flush=True)

    def frames(self, buffer) -> None:
        ts, values = buffer.last_seconds(2 * config.HOP_S + 1.0)
        new = ts > self.last_ts
        for t, row in zip(ts[new], values[new]):
            self.raw.write(f"{t:.6f}," + ",".join("" if np.isnan(v) else f"{v:.6g}" for v in row) + "\n")
        if new.any():
            self.last_ts = ts[new][-1]
        self.raw.flush()

    def window(self, t, stream_ts, hands, frames, p, word) -> None:
        i = int(p.argmax())
        self.pred.write(f"{t:.3f},{stream_ts:.6f},{'+'.join(hands) or 'none'},{frames},"
                        + ",".join(f"{v:.4f}" for v in p)
                        + f",{self.classes[i]},{p[i]:.4f},{word or ''}\n")
        self.pred.flush()

    def close(self) -> None:
        self.pred.close()
        self.raw.close()


def live(model, decoder_kw: dict, trace: bool, log_dir: str | None = None, adapt: float = 0.0) -> None:
    from rokoko_stream import COLUMNS, GloveStreamReader   # data-capture; Windows + Rokoko Device SDK
    feats = LiveFeatures()
    dec, tr = VoteDecoder(model.classes, **decoder_kw), Transcript()
    n = realtime.window_frames()
    log = SessionLog(log_dir, model.classes, COLUMNS) if log_dir else None
    ad, last_ts = (Adapter(model, adapt) if adapt else None), None
    with GloveStreamReader() as reader:
        print(f"Listening for {', '.join(c for c in model.classes if c != config.IDLE_LABEL)}"
              " ... Ctrl+C to stop", flush=True)
        t0, warned = time.time(), set()
        try:
            while True:
                time.sleep(config.HOP_S)
                if log:
                    log.frames(reader.buffer)
                hands = reader.connected_hands
                state = tuple(hands)
                if len(hands) < 2 and state not in warned:
                    print(f"  (gloves connected: {hands or 'none'}"
                          + (f"; error: {reader.last_error}" if reader.last_error else "") + ")",
                          flush=True)
                    warned.add(state)
                x = feats(reader.buffer, config.WINDOW_S)
                if len(x) < config.MIN_FILL * n:
                    continue
                if ad is not None:   # push only the frames that arrived since the last hop
                    now = float(reader.buffer.latest(1)[0][-1])
                    k = len(x) if last_ts is None else int(round((now - last_ts) * config.FEATURE_RATE_HZ))
                    ad.push(x[len(x) - min(max(k, 0), len(x)):])
                    last_ts = now
                    if ad.ready and not getattr(ad, "announced", False):
                        print("  (session alignment active)", flush=True)
                        ad.announced = True
                    x = ad(x)
                p = model.predict(realtime.fit_window(x)[None])[0]
                t = time.time() - t0
                if trace:
                    i = int(p.argmax())
                    print(f"   {t:6.2f}s  {model.classes[i]:<10}{p[i]:.2f}", flush=True)
                w = dec.step(p)
                if log:
                    log.window(t, float(reader.buffer.latest(1)[0][-1]), hands, len(x), p, w)
                if w:
                    show(t, w, tr)
        except KeyboardInterrupt:
            pass
        finally:
            if log:
                log.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--replay", nargs="+", metavar="CSV", help="offline replay instead of the gloves")
    ap.add_argument("--trace", action="store_true", help="print the top classes of every window")
    ap.add_argument("--confidence", type=float, default=None, help=f"default {config.CONFIDENCE}")
    ap.add_argument("--votes", type=int, default=None, help=f"default {config.VOTES}")
    ap.add_argument("--hop", type=float, default=None, help=f"seconds, default {config.HOP_S}")
    ap.add_argument("--no-class-check", action="store_true",
                    help="allow a model whose classes differ from the data/ folders")
    ap.add_argument("--log", nargs="?", const=str(ROOT.parent / "takes" / "live_logs"), default=None,
                    metavar="DIR", help="live only: log per-window predictions and the raw stream "
                                        "(default dir: takes/live_logs)")
    ap.add_argument("--adapt", nargs="?", type=float, const=60.0, default=0.0, metavar="SECONDS",
                    help="align features to the training distribution using the last SECONDS "
                         "(default 60) of the stream; active after 10 s (signrec.live.Adapter)")
    a = ap.parse_args()
    if a.hop:
        config.HOP_S = a.hop
    model = LiveSignModel.load(a.model, check_classes=not a.no_class_check)
    kw = {"confidence": a.confidence, "votes": a.votes}
    print(f"model {a.model}: {model.classes}; confidence {a.confidence or config.CONFIDENCE}, "
          f"votes {a.votes or config.VOTES}, hop {config.HOP_S}s")
    if a.replay:
        for path in a.replay:
            replay(model, path, kw, a.trace, a.adapt)
    else:
        live(model, kw, a.trace, a.log, a.adapt)


if __name__ == "__main__":
    main()
