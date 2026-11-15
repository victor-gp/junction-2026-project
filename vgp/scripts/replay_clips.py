"""Feed recorded Studio clips to a running clip_demo.py, as if Studio had just exported them.

Fallback for the demo when new recordings can't be used (e.g. Studio saves takes
without glove motion). Each clip is copied into the --watch folder under a fresh
name, one every --interval seconds; clip_demo.py classifies and speaks it.

Honest pairing: run clip_demo.py with the hold-out model and replay hold-out clips,
so every clip is one the model never trained on:

    py scripts\\clip_demo.py --watch <DIR> --gui --model results\\live_cnn_holdout.npz
    py scripts\\replay_clips.py --watch <DIR>                         # all 36 hold-out clips
    py scripts\\replay_clips.py --watch <DIR> --words Hello "Thank you" Okay --per-word 1
    py scripts\\replay_clips.py --watch <DIR> --files "..\\data\\No\\Clip_44.csv"

Copied files are named replay_<time>_<Label>_<clip>.csv; delete them afterwards if you like.
"""
import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", required=True, help="folder clip_demo.py is watching")
    ap.add_argument("--list", default=str(ROOT / "results" / "live_holdout.json"),
                    help='JSON {"clips": ["<Label>/<file>", ...]} relative to data/ (default: hold-out set)')
    ap.add_argument("--files", nargs="+", metavar="CSV", help="replay these files instead of --list")
    ap.add_argument("--words", nargs="+", metavar="LABEL", help="only these labels (folder names)")
    ap.add_argument("--per-word", type=int, default=0, help="at most N clips per label (0 = all)")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between clips (default 5)")
    ap.add_argument("--shuffle", action="store_true", help="random order")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    watch = Path(a.watch)
    if not watch.is_dir():
        sys.exit(f"--watch folder not found: {watch}")

    if a.files:
        clips = [Path(f) for f in a.files]
    else:
        clips = [DATA / c for c in json.loads(Path(a.list).read_text())["clips"]]
    if a.words:
        wanted = set(a.words)
        clips = [c for c in clips if c.parent.name in wanted]
    if a.per_word:
        seen: dict[str, int] = {}
        kept = []
        for c in clips:
            n = seen.get(c.parent.name, 0)
            if n < a.per_word:
                kept.append(c)
                seen[c.parent.name] = n + 1
        clips = kept
    if a.shuffle:
        random.Random(a.seed).shuffle(clips)
    missing = [c for c in clips if not c.is_file()]
    if missing:
        sys.exit(f"missing clips: {[str(m) for m in missing[:5]]}")
    if not clips:
        sys.exit("no clips selected")

    print(f"replaying {len(clips)} clips into {watch.resolve()} every {a.interval:g} s (Ctrl+C to stop)")
    try:
        for i, c in enumerate(clips):
            name = f"replay_{time.strftime('%H%M%S')}_{c.parent.name.replace(' ', '')}_{c.stem}.csv"
            tmp = watch / (name + ".part")          # not *.csv, so the watcher ignores it mid-copy
            shutil.copyfile(c, tmp)
            tmp.replace(watch / name)
            print(f"[{i + 1}/{len(clips)}] {c.parent.name}/{c.name}  ->  {name}", flush=True)
            if i < len(clips) - 1:
                time.sleep(a.interval)
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
