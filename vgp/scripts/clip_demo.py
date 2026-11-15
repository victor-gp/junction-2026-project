"""Clip demo: sign in Rokoko Studio, export the take as CSV, hear and see the word.

Watches Studio's export folder. Each new (or re-exported) CSV is classified as a whole
clip: 3 s windows of the 114 live features, window probabilities averaged, argmax.
This is the same rule as train_live.py --eval's clip accuracy. Idle, or a top
probability below --threshold, shows "not sure" and says nothing.

From vgp/ (numpy + pandas only; --gui needs tkinter, bundled with python.org Python):
    python scripts/clip_demo.py --watch "C:/Users/<you>/Documents/Rokoko Studio/Exports"
    python scripts/clip_demo.py --watch <dir> --gui           # big-text window for the video
        (each recognised word also pops up as an always-on-top overlay that fades after
         --popup-seconds (4), without taking keyboard focus from Studio; click it to dismiss.
         --popup-pos bottom|top|center|top-right|bottom-right; --no-popup for the old
         behaviour (main window comes to the front); --fullscreen / F11, --on-top)
    python scripts/clip_demo.py --watch <dir> --overlay-only  # popups only over Studio
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

# -- popup overlay ------------------------------------------------------------------

POPUP_CARD, POPUP_ACCENT, POPUP_NEUTRAL = "#111418", "#2ecc71", "#8ab4f8"
POPUP_POSITIONS = ("bottom", "top", "center", "top-right", "bottom-right")


def work_area(root) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the primary screen minus the taskbar."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            r = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(r), 0):  # SPI_GETWORKAREA
                return r.left, r.top, r.right, r.bottom
        except Exception:
            pass
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


def make_overlay_window(win) -> None:
    """Windows: never take keyboard focus (WS_EX_NOACTIVATE), no taskbar button
    (WS_EX_TOOLWINDOW), and sit above all normal windows without activating."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        u.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        u.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
        u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        hwnd = wintypes.HWND(int(win.wm_frame(), 16))
        ex = u.GetWindowLongW(hwnd, -20)                                  # GWL_EXSTYLE
        u.SetWindowLongW(hwnd, -20, ex | 0x08000000 | 0x00000080)         # NOACTIVATE | TOOLWINDOW
        flags = 0x0001 | 0x0002 | 0x0010 | 0x0040                         # NOSIZE|NOMOVE|NOACTIVATE|SHOW
        u.SetWindowPos(hwnd, wintypes.HWND(-1), 0, 0, 0, 0, flags)        # HWND_TOPMOST
    except Exception:
        pass


class _Toast:
    """One popup card: fades in, holds `seconds` (with a countdown bar), fades out."""
    STEP_MS, FADE_IN, FADE_OUT = 30, 0.18, 0.35

    def __init__(self, stack: "PopupStack", title: str, subtitle: str, meta: str,
                 conf: float | None, accent: str, seconds: float):
        import tkinter as tk
        self.stack, self.seconds, self.dead = stack, seconds, False
        l, t, r, b = work_area(stack.root)
        sw, sh = r - l, b - t
        pad = max(10, int(sh * 0.018))
        px_title, px_sub, px_meta = max(28, int(sh * 0.075)), max(14, int(sh * 0.032)), max(10, int(sh * 0.017))

        self.top = top = tk.Toplevel(stack.root)
        top.withdraw()
        top.overrideredirect(True)
        top.attributes("-topmost", True)
        top.attributes("-alpha", 0.0)
        top.configure(bg=accent)
        tk.Frame(top, bg=accent, width=max(6, sh // 140)).pack(side="left", fill="y")
        body = tk.Frame(top, bg=POPUP_CARD, padx=pad * 2, pady=pad)
        body.pack(side="left", fill="both", expand=True)
        row = tk.Frame(body, bg=POPUP_CARD)
        row.pack(fill="x")
        tk.Label(row, text=title, font=("Segoe UI", -px_title, "bold"), fg="#ffffff",
                 bg=POPUP_CARD).pack(side="left")
        if conf is not None:
            tk.Label(row, text=f"{conf:.0%}", font=("Segoe UI", -int(px_title * 0.45), "bold"),
                     fg=accent, bg=POPUP_CARD).pack(side="right", anchor="s", padx=(pad * 3, 0),
                                                     pady=(0, int(px_title * 0.12)))
        if subtitle:
            tk.Label(body, text=subtitle, font=("Segoe UI", -px_sub), fg=POPUP_NEUTRAL,
                     bg=POPUP_CARD, anchor="w").pack(fill="x")
        if meta:
            tk.Label(body, text=meta, font=("Consolas", -px_meta), fg="#7a7f87",
                     bg=POPUP_CARD, anchor="w").pack(fill="x", pady=(pad // 2, 0))
        self.bar = tk.Canvas(body, height=max(4, sh // 220), bg="#2a2f36", highlightthickness=0)
        self.bar.pack(fill="x", pady=(pad, 0))
        self.bar_rect = self.bar.create_rectangle(0, 0, 0, 0, fill=accent, width=0)

        top.update_idletasks()
        self.w = max(top.winfo_reqwidth(), int(sw * 0.24))
        self.h = top.winfo_reqheight()
        for wdg in [top, *self._descendants(top)]:          # click anywhere to dismiss
            wdg.bind("<Button-1>", lambda e: self.dismiss())

        self.phase, self.t0, self.alpha, self.a_start = "in", time.monotonic(), 0.0, 0.0

    def _descendants(self, w):
        for c in w.winfo_children():
            yield c
            yield from self._descendants(c)

    def place(self, x: int, y: int) -> None:
        self.top.geometry(f"{self.w}x{self.h}+{x}+{y}")

    def start(self) -> None:
        self.top.deiconify()
        self.top.update_idletasks()
        make_overlay_window(self.top)
        self._tick()

    def dismiss(self) -> None:
        if not self.dead and self.phase != "out":
            self.phase, self.t0, self.a_start = "out", time.monotonic(), self.alpha

    def _set_alpha(self, a: float) -> None:
        self.alpha = a
        self.top.attributes("-alpha", a)

    def _tick(self) -> None:
        if self.dead:
            return
        dt = time.monotonic() - self.t0
        if self.phase == "in":
            f = min(1.0, dt / self.FADE_IN)
            self._set_alpha(f * self.stack.alpha)
            if f >= 1.0:
                self.phase, self.t0 = "hold", time.monotonic()
        elif self.phase == "hold":
            left = max(0.0, 1.0 - dt / self.seconds)
            self.bar.coords(self.bar_rect, 0, 0, int(self.bar.winfo_width() * left),
                            int(self.bar.cget("height")))
            if left <= 0.0:
                self.phase, self.t0, self.a_start = "out", time.monotonic(), self.alpha
        else:
            f = 1.0 - dt / self.FADE_OUT
            if f <= 0.0:
                self.dead = True
                self.top.destroy()
                self.stack.remove(self)
                return
            self._set_alpha(f * self.a_start)
        self.top.after(self.STEP_MS, self._tick)


class PopupStack:
    """Popup overlays, one per recognised word. Borderless, always on top, never steal
    keyboard focus (Studio keeps it), fade out after `seconds`. Words that arrive in
    quick succession stack, newest nearest the anchor; at most `max_visible` at once."""

    def __init__(self, root, seconds: float = 4.0, position: str = "bottom",
                 alpha: float = 0.94, max_visible: int = 3):
        self.root, self.seconds, self.position = root, seconds, position
        self.alpha, self.max_visible = alpha, max_visible
        self.toasts: list[_Toast] = []

    def show(self, title: str, subtitle: str = "", meta: str = "", conf: float | None = None,
             accent: str = POPUP_ACCENT, seconds: float | None = None) -> None:
        t = _Toast(self, title, subtitle, meta, conf, accent, seconds or self.seconds)
        self.toasts.insert(0, t)
        for old in self.toasts[self.max_visible:]:
            old.dismiss()
        self.layout()
        t.start()

    def remove(self, t: _Toast) -> None:
        if t in self.toasts:
            self.toasts.remove(t)
            self.layout()

    def layout(self) -> None:
        l, t, r, b = work_area(self.root)
        margin, gap = int((b - t) * 0.04), int((b - t) * 0.012)
        offset = 0
        for toast in self.toasts:
            w, h = toast.w, toast.h
            x = r - margin - w if self.position.endswith("right") else l + (r - l - w) // 2
            if self.position.startswith("top"):
                y = t + margin + offset                      # older ones slide down
            elif self.position == "center":
                y = t + (b - t - h) // 2 - offset            # older ones slide up
            else:
                y = b - margin - h - offset                  # bottom: older ones slide up
            toast.place(x, y)
            offset += h + gap


def run_gui(a, model, speaker):
    import signal
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
    popups = None if a.no_popup else PopupStack(root, a.popup_seconds, a.popup_pos, a.popup_alpha)

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
        if popups:                               # the popup is the notification ...
            popups.show(english(word), f"FinSL: {finsl(word)}",
                        f"{name}  ·  {time.strftime('%H:%M:%S')}", conf)
        elif not a.no_raise:                     # ... otherwise bring the main window forward
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
    signal.signal(signal.SIGINT, lambda *_: root.after(0, root.destroy))   # Ctrl+C in the console
    if a.overlay_only:
        root.iconify()                           # popups only; close the taskbar window to quit
        root.after(300, lambda: popups.show("Ready", "sign, then export", f"watching {a.watch}",
                                            accent=POPUP_NEUTRAL, seconds=3.0))
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
                    help="--gui --no-popup: don't bring the window to the front on a recognised word")
    ap.add_argument("--no-popup", action="store_true",
                    help="--gui: no popup overlay; bring the main window forward instead")
    ap.add_argument("--popup-seconds", type=float, default=4.0,
                    help="--gui: how long each word popup stays on screen (default 4)")
    ap.add_argument("--popup-pos", choices=POPUP_POSITIONS, default="bottom",
                    help="--gui: where popups appear on the primary screen (default bottom)")
    ap.add_argument("--popup-alpha", type=float, default=0.94,
                    help="--gui: popup opacity 0..1 (default 0.94)")
    ap.add_argument("--overlay-only", action="store_true",
                    help="popups only: implies --gui, main window starts minimised")
    ap.add_argument("--existing", action="store_true", help="also classify CSVs already in --watch")
    ap.add_argument("--poll", type=float, default=0.5, help="seconds between folder scans")
    ap.add_argument("--settle", type=float, default=1.0,
                    help="seconds a file must stay unchanged before it is read")
    ap.add_argument("--quiet", action="store_true", help="--files/--list: summary only")
    ap.add_argument("--no-class-check", action="store_true",
                    help="run even if data/ folders differ from the model's classes")
    a = ap.parse_args()
    if a.overlay_only:
        if a.no_popup:
            ap.error("--overlay-only needs the popups; drop --no-popup")
        a.gui = True

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
