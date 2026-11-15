"""Clip demo: sign in Rokoko Studio, export the take as CSV, hear and see the word.

Watches Studio's export folder. Each new (or re-exported) CSV is classified as a whole
clip: 3 s windows of the 114 live features, window probabilities averaged, argmax.
This is the same rule as train_live.py --eval's clip accuracy. Idle, or a top
probability below --threshold, shows "not sure" and says nothing.

From vgp/ (numpy + pandas only; --gui needs tkinter, bundled with python.org Python):
    python scripts/clip_demo.py --watch "C:/Users/<you>/Documents/Rokoko Studio/Exports"
    python scripts/clip_demo.py --watch <dir> --gui           # big-text window for the video
    python scripts/clip_demo.py --files ../data/No/Clip_150.csv ...       # classify, exit
    python scripts/clip_demo.py --list results/live_holdout.json \\
        --model results/live_cnn_holdout.npz                             # accuracy + threshold sweep

Voice: cached ElevenLabs clips from scripts/make_audio.py, else system TTS (--no-voice
to mute). Only files that appear after start-up are classified (--existing for all).
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from signrec import voice  # noqa: E402
from signrec.live import (DEFAULT_MODEL, GLOSS, LiveSignModel, config,  # noqa: E402
                          file_features, realtime)

DEFAULT_THRESHOLD = 0.5


# -- classification ---------------------------------------------------------------

def clip_probs(model: LiveSignModel, path) -> np.ndarray:
    feats = file_features(path)
    if len(feats) < 2:
        raise ValueError("clip has fewer than 2 frames")
    wins = np.stack(realtime.clip_windows(feats)).astype(np.float32)
    return model.predict(wins).mean(0)


def decide(model: LiveSignModel, probs: np.ndarray, threshold: float) -> str | None:
    """The word to say, or None for Idle / not confident enough."""
    label = model.classes[int(probs.argmax())]
    if label == config.IDLE_LABEL or probs.max() < threshold:
        return None
    return label


def describe(model: LiveSignModel, probs: np.ndarray, k: int = 3) -> str:
    top = np.argsort(probs)[::-1][:k]
    return "  ".join(f"{model.classes[i]} {probs[i]:.2f}" for i in top)


# -- offline evaluation -----------------------------------------------------------

def evaluate(model: LiveSignModel, paths: list[Path], threshold: float, quiet: bool) -> None:
    rows = []
    for p in paths:
        truth = p.parent.name
        probs = clip_probs(model, p)
        rows.append((truth, probs))
        if not quiet:
            word = decide(model, probs, threshold) or "-"
            mark = "ok" if (word == truth or (word == "-" and truth == config.IDLE_LABEL)) else "XX"
            print(f"{mark}  {truth + '/' + p.name:<28} -> {word:<10} | {describe(model, probs)}")
    known = [(t, pr) for t, pr in rows if t in model.classes]
    if not known:
        return
    argmax_acc = np.mean([model.classes[int(pr.argmax())] == t for t, pr in known])
    print(f"\n{len(known)} labelled clips, argmax clip accuracy {argmax_acc:.0%}")
    print("(sign clips: right / wrong word / silent;  Idle clips: silent / spoke)")
    print("threshold   right   wrong   silent   precision  |  Idle silent")
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        right = wrong = silent = idle_ok = idle_n = 0
        for t, pr in known:
            w = decide(model, pr, th)
            if t == config.IDLE_LABEL:
                idle_n += 1
                idle_ok += w is None
            elif w is None:
                silent += 1
            elif w == t:
                right += 1
            else:
                wrong += 1
        spoke = right + wrong + (idle_n - idle_ok)
        mark = "  <-" if abs(th - threshold) < 1e-9 else ""
        print(f"  {th:.1f}       {right:4d}   {wrong:4d}   {silent:5d}     {right / max(spoke, 1):6.0%}    "
              f"|  {idle_ok}/{idle_n}{mark}")


# -- watching -----------------------------------------------------------------------

class ExportWatcher:
    """Polls `folder` (recursively) for .csv files. A file is ready once its size and
    mtime have not changed for `settle` seconds (Studio may still be writing it). A
    re-export to the same name (new mtime) is classified again."""

    def __init__(self, folder: Path, settle: float = 1.0, existing: bool = False):
        self.folder = Path(folder)
        self.settle = settle
        self.pending: dict[Path, tuple[tuple[int, float], float]] = {}   # path -> (stat, since)
        self.done: dict[Path, tuple[int, float]] = {}
        if not existing:
            for p, st in self._scan().items():
                self.done[p] = st

    def _scan(self) -> dict[Path, tuple[int, float]]:
        out = {}
        for root, _, files in os.walk(self.folder):
            for f in files:
                if f.lower().endswith(".csv"):
                    p = Path(root) / f
                    try:
                        s = p.stat()
                    except OSError:
                        continue
                    out[p] = (s.st_size, s.st_mtime)
        return out

    def poll(self) -> list[Path]:
        now, ready = time.monotonic(), []
        for p, st in self._scan().items():
            if self.done.get(p) == st or st[0] == 0:
                continue
            prev = self.pending.get(p)
            if prev is None or prev[0] != st:
                self.pending[p] = (st, now)
            elif now - prev[1] >= self.settle:
                del self.pending[p]
                self.done[p] = st
                ready.append(p)
        return sorted(ready, key=lambda q: self.done[q][1])

    def retry(self, p: Path, max_tries: int = 5) -> bool:
        """Read failed (on Windows, Studio may still hold the file locked): look at it
        again on a later poll. False once it has failed max_tries times."""
        self.failures = getattr(self, "failures", {})
        self.failures[p] = self.failures.get(p, 0) + 1
        if self.failures[p] >= max_tries:
            return False
        self.done.pop(p, None)
        return True


def english(label: str) -> str:
    return GLOSS.get(label, (None, label))[1]


def finsl(label: str) -> str:
    return GLOSS.get(label, (label.upper(), label))[0]


def watch_loop(a, model, speaker, events: queue.Queue | None, stop: threading.Event):
    w = ExportWatcher(Path(a.watch), settle=a.settle, existing=a.existing)
    print(f"watching {Path(a.watch).resolve()} for Studio CSV exports (Ctrl+C to stop)", flush=True)
    while not stop.is_set():
        for p in w.poll():
            t0 = time.perf_counter()
            try:
                probs = clip_probs(model, p)
            except Exception as e:
                if w.retry(p):
                    continue
                print(f"[{time.strftime('%H:%M:%S')}] {p.name}: could not read ({e})", flush=True)
                if events:
                    events.put(("error", p.name, str(e)))
                continue
            word = decide(model, probs, a.threshold)
            ms = (time.perf_counter() - t0) * 1000
            stamp = time.strftime("%H:%M:%S")
            if word:
                print(f"[{stamp}] {p.name}:  >>> {english(word).upper()}  ({finsl(word)})  "
                      f"{probs.max():.2f}   | {describe(model, probs)}  [{ms:.0f} ms]", flush=True)
                speaker.speak(word, voice.spoken_text(word, english(word), a.lang))
            else:
                print(f"[{stamp}] {p.name}:  (not sure)   | {describe(model, probs)}  [{ms:.0f} ms]",
                      flush=True)
            if events:
                events.put(("word" if word else "unsure", p.name, word, float(probs.max()),
                            describe(model, probs)))
        stop.wait(a.poll)


def run_gui(a, model, speaker):
    import tkinter as tk

    events: queue.Queue = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=watch_loop, args=(a, model, speaker, events, stop), daemon=True).start()

    if sys.platform == "win32":     # crisp text on high-DPI laptop screens
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    root.title("Glove sign recognition")
    root.configure(bg="#111")
    root.geometry("1000x560")
    big = tk.Label(root, text="sign, then export", font=("Segoe UI", 96, "bold"), fg="#eee", bg="#111")
    big.pack(expand=True, fill="both")
    gl = tk.Label(root, text="", font=("Segoe UI", 36), fg="#8ab4f8", bg="#111")
    gl.pack()
    small = tk.Label(root, text=f"watching {a.watch}", font=("Consolas", 14), fg="#888", bg="#111")
    small.pack(pady=(10, 4))
    hist = tk.Label(root, text="", font=("Segoe UI", 18), fg="#bbb", bg="#111")
    hist.pack(pady=(0, 16))
    said: list[str] = []

    def tick():
        try:
            while True:
                ev = events.get_nowait()
                if ev[0] == "word":
                    _, name, word, conf, desc = ev
                    big.config(text=english(word), fg="#eee")
                    gl.config(text=f"FinSL: {finsl(word)}   ·   {conf:.0%}")
                    small.config(text=f"{name}   |   {desc}")
                    said.append(english(word))
                    hist.config(text="  ·  ".join(said[-8:]))
                elif ev[0] == "unsure":
                    _, name, _, conf, desc = ev
                    big.config(text="not sure", fg="#777")
                    gl.config(text="")
                    small.config(text=f"{name}   |   {desc}")
                else:
                    big.config(text="read error", fg="#c66")
                    small.config(text=f"{ev[1]}: {ev[2]}")
        except queue.Empty:
            pass
        root.after(100, tick)

    root.bind("<Escape>", lambda e: root.destroy())
    root.after(100, tick)
    try:
        root.mainloop()
    finally:
        stop.set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--watch", metavar="DIR", help="Studio export folder to watch")
    src.add_argument("--files", nargs="+", metavar="CSV", help="classify these clips and exit")
    src.add_argument("--list", metavar="JSON", help='clips {"clips": ["<Label>/<file>", ...]} '
                     "relative to data/ (e.g. results/live_holdout.json); classify and exit")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"min mean probability to say a word (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--lang", choices=voice.LANGS, default="en",
                    help="spoken language: en (audio/) or fi (audio/fi/, e.g. 'Kiitos')")
    ap.add_argument("--gui", action="store_true", help="big-text window (tkinter)")
    ap.add_argument("--existing", action="store_true", help="also classify CSVs already in --watch")
    ap.add_argument("--poll", type=float, default=0.5, help="seconds between folder scans")
    ap.add_argument("--settle", type=float, default=1.0,
                    help="seconds a file must stay unchanged before it is read")
    ap.add_argument("--quiet", action="store_true", help="--files/--list: summary only")
    ap.add_argument("--no-class-check", action="store_true",
                    help="run even if data/ folders differ from the model's classes")
    a = ap.parse_args()

    model = LiveSignModel.load(a.model, check_classes=not a.no_class_check)
    print(f"model {a.model}: {len(model.classes)} classes {model.classes}; "
          f"trained on {sum(model.meta.trained_on.values())} clips; threshold {a.threshold}", flush=True)

    if a.files or a.list:
        if a.list:
            clips = json.loads(Path(a.list).read_text())["clips"]
            paths = [Path(config.DATA_DIR) / c for c in clips]
        else:
            paths = [Path(f) for f in a.files]
        evaluate(model, paths, a.threshold, a.quiet)
        return

    if not Path(a.watch).is_dir():
        sys.exit(f"--watch folder not found: {a.watch}")
    speaker = voice.Speaker(voice.lang_dir(a.lang), enabled=not a.no_voice)
    if not a.no_voice:
        srcs = {speaker.source(c) for c in model.classes if c != config.IDLE_LABEL}
        print(f"voice ({a.lang}): {', '.join(sorted(srcs))}", flush=True)
    if a.gui:
        run_gui(a, model, speaker)
        return
    stop = threading.Event()
    try:
        watch_loop(a, model, speaker, None, stop)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
