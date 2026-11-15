"""Clip demo: sign in Rokoko Studio, export the take as CSV, hear and see the word.

Watches Studio's export folder. Each new (or re-exported) CSV is classified as a whole
clip: 3 s windows of the 114 live features, window probabilities averaged, argmax.
This is the same rule as train_live.py --eval's clip accuracy. Idle, or a top
probability below --threshold, shows "not sure" and says nothing.

From vgp/ (numpy + pandas only; --gui needs tkinter, bundled with python.org Python):
    python scripts/clip_demo.py --watch "C:/Users/<you>/Documents/Rokoko Studio/Exports"
    python scripts/clip_demo.py --watch <dir> --gui           # big-text window for the video
        (comes to the front on each recognised word without taking keyboard focus;
         --fullscreen / F11 for a projector, --on-top to pin it, --no-raise to never raise)
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
                if events is None:      # GUI mode: the window speaks when it shows the word
                    speaker.speak(word, english(word))
            else:
                print(f"[{stamp}] {p.name}:  (not sure)   | {describe(model, probs)}  [{ms:.0f} ms]",
                      flush=True)
            if events:
                events.put(("word" if word else "unsure", p.name, word, float(probs.max()),
                            describe(model, probs)))
        stop.wait(a.poll)


def raise_without_focus(root, keep_on_top: bool) -> None:
    """Put the window in front of other windows without activating it, so keyboard
    focus stays where it is (e.g. in Rokoko Studio). Windows: SetWindowPos with
    SWP_NOACTIVATE (TOPMOST then NOTOPMOST = to the top of the normal stack)."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            u = ctypes.windll.user32
            u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_uint]
            hwnd = wintypes.HWND(int(root.wm_frame(), 16))
            if u.IsIconic(hwnd):
                u.ShowWindow(hwnd, 4)                     # SW_SHOWNOACTIVATE
            flags = 0x0001 | 0x0002 | 0x0010 | 0x0040    # NOSIZE | NOMOVE | NOACTIVATE | SHOWWINDOW
            u.SetWindowPos(hwnd, wintypes.HWND(-1), 0, 0, 0, 0, flags)          # HWND_TOPMOST
            if not keep_on_top:
                u.SetWindowPos(hwnd, wintypes.HWND(-2), 0, 0, 0, 0, flags)      # HWND_NOTOPMOST
            return
        except Exception:
            pass
    root.lift()
    root.attributes("-topmost", True)
    if not keep_on_top:
        root.after(50, lambda: root.attributes("-topmost", False))


BG, FLASH = "#000000", "#0b4d24"


def run_gui(a, model, speaker):
    import tkinter as tk
    import tkinter.font as tkfont

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
    root.configure(bg=BG)
    root.geometry("1200x700")
    if a.fullscreen:
        root.attributes("-fullscreen", True)
    if a.on_top:
        root.attributes("-topmost", True)

    # Fonts sized in pixels (negative size) and rescaled with the window, see fit().
    f_word = tkfont.Font(family="Segoe UI", size=-120, weight="bold")
    f_conf = tkfont.Font(family="Segoe UI", size=-56, weight="bold")
    f_gloss = tkfont.Font(family="Segoe UI", size=-40)
    f_hist = tkfont.Font(family="Segoe UI", size=-26)
    f_status = tkfont.Font(family="Consolas", size=-18)

    word_l = tk.Label(root, text="sign, then export", font=f_word, fg="#666", bg=BG)
    word_l.pack(expand=True, fill="both")
    conf_row = tk.Frame(root, bg=BG)
    conf_row.pack()
    conf_l = tk.Label(conf_row, text="", font=f_conf, fg="#7ee2a8", bg=BG)
    conf_l.pack(side="left", padx=(0, 24))
    bar = tk.Canvas(conf_row, width=400, height=36, bg="#222", highlightthickness=0)
    gloss_l = tk.Label(root, text="", font=f_gloss, fg="#8ab4f8", bg=BG)
    gloss_l.pack(pady=(8, 0))
    hist_l = tk.Label(root, text="", font=f_hist, fg="#aaa", bg=BG)
    hist_l.pack(pady=(18, 4))
    status_l = tk.Label(root, text=f"watching {a.watch}", font=f_status, fg="#777", bg=BG)
    status_l.pack(pady=(0, 14))
    widgets = [root, word_l, conf_row, conf_l, gloss_l, hist_l, status_l]
    said: list[str] = []
    flash_job = [None]

    def fit(_=None):
        w, h = max(root.winfo_width(), 200), max(root.winfo_height(), 200)
        text = word_l.cget("text") or " "
        px = int(h * 0.30)                       # word: ~30% of the window height ...
        f_word.configure(size=-px)
        while px > 24 and f_word.measure(text) > w * 0.94:   # ... shrunk until it fits the width
            px = int(px * 0.92)
            f_word.configure(size=-px)
        f_conf.configure(size=-max(20, int(h * 0.085)))
        f_gloss.configure(size=-max(16, int(h * 0.055)))
        f_hist.configure(size=-max(12, int(h * 0.036)))
        f_status.configure(size=-max(10, int(h * 0.025)))
        bar.configure(width=int(w * 0.35), height=max(14, int(h * 0.05)))

    def draw_bar(conf: float):
        bar.update_idletasks()
        bw, bh = int(bar.cget("width")), int(bar.cget("height"))
        bar.delete("all")
        bar.create_rectangle(0, 0, int(bw * conf), bh, fill="#2ecc71", width=0)
        tx = int(bw * a.threshold)               # tick at the speaking threshold
        bar.create_line(tx, 0, tx, bh, fill="#ddd", width=2)

    def set_bg(color: str):
        for wdg in widgets:
            wdg.configure(bg=color)

    def flash():
        if flash_job[0]:
            root.after_cancel(flash_job[0])
        set_bg(FLASH)
        flash_job[0] = root.after(700, lambda: set_bg(BG))

    def show_word(name, word, conf, desc):
        word_l.config(text=english(word), fg="#ffffff")
        conf_l.config(text=f"{conf:.0%}")
        if not bar.winfo_ismapped():
            bar.pack(side="left")
        fit()
        draw_bar(conf)
        gloss_l.config(text=f"FinSL: {finsl(word)}")
        said.append(english(word))
        hist_l.config(text="  ·  ".join(said[-6:]))
        status_l.config(text=f"{name}   |   {desc}", fg="#777")
        flash()
        if not a.no_raise:
            raise_without_focus(root, a.on_top)
        root.update_idletasks()                  # the word is on screen before it is heard
        speaker.speak(word, english(word))

    def show_quiet(text, fg):
        """Idle / not confident / read error: no big word, no flash, no raise, no voice."""
        word_l.config(text="·  ·  ·", fg="#333")
        conf_l.config(text="")
        bar.pack_forget()
        gloss_l.config(text="")
        status_l.config(text=text, fg=fg)
        fit()

    def tick():
        try:
            while True:
                ev = events.get_nowait()
                if ev[0] == "word":
                    _, name, word, conf, desc = ev
                    show_word(name, word, conf, desc)
                elif ev[0] == "unsure":
                    _, name, _, conf, desc = ev
                    show_quiet(f"not sure, nothing said   |   {name}   |   {desc}", "#c9a227")
                else:
                    show_quiet(f"could not read {ev[1]}: {ev[2]}", "#d66")
        except queue.Empty:
            pass
        root.after(100, tick)

    root.bind("<Configure>", lambda e: fit() if e.widget is root else None)
    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("<F11>", lambda e: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
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
    ap.add_argument("--gui", action="store_true", help="big-text window (tkinter)")
    ap.add_argument("--fullscreen", action="store_true", help="--gui: start full screen (F11 toggles)")
    ap.add_argument("--on-top", action="store_true", help="--gui: keep the window above all others")
    ap.add_argument("--no-raise", action="store_true",
                    help="--gui: don't bring the window to the front on a recognised word")
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
    speaker = voice.Speaker(enabled=not a.no_voice)
    if not a.no_voice:
        srcs = {speaker.source(c) for c in model.classes if c != config.IDLE_LABEL}
        print(f"voice: {', '.join(sorted(srcs))}", flush=True)
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
