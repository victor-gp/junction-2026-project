"""Does a Studio CSV export actually contain hand motion? Compares it with the training data.

A take whose glove data was not saved exports as one frozen pose: wrist travel of
hundredths of a millimetre and finger bends that change by < 0.1 deg, against
~0.2 m and ~15 deg in the training clips. clip_demo.py then answers "not sure" or a
wrong word, and no model change can fix that.

    py scripts\\check_motion.py "<Studio export folder>"          # every CSV in the folder
    py scripts\\check_motion.py path\\to\\take.csv other.csv

Columns: frames; per hand (L/R) wrist travel in m (bounding-box diagonal of the hand
position), mean range of knuckle bends in deg, and the number of constant columns.
Verdict FROZEN when both hands move < 1 mm and bend < 0.2 deg.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
FLEX = re.compile(r"(Metacarpophalangeal|ProximalInterphalangeal)_flexion$")


def stats(path) -> dict:
    d = pd.read_csv(path)
    out = {"frames": len(d)}
    for h in ("Left", "Right"):
        pos = d[[f"{h}Hand_position_{a}" for a in "xyz"]].to_numpy(float)
        flex = [c for c in d.columns if c.startswith(h) and FLEX.search(c)]
        cols = [c for c in d.columns if c.startswith(h)]
        out[f"{h[0]}_move_m"] = float(np.linalg.norm(pos.max(0) - pos.min(0)))
        out[f"{h[0]}_bend_deg"] = float((d[flex].max() - d[flex].min()).mean())
        out[f"{h[0]}_const"] = int((d[cols].std() < 1e-6).sum())
    frozen = all(out[f"{h}_move_m"] < 1e-3 and out[f"{h}_bend_deg"] < 0.2 for h in "LR")
    out["verdict"] = "FROZEN" if frozen else "moving"
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        paths += sorted(p.rglob("*.csv")) if p.is_dir() else [p]
    rows = {}
    for p in paths:
        try:
            rows[p.name] = stats(p)
        except Exception as e:                      # not a Studio hand CSV
            rows[p.name] = {"verdict": f"unreadable ({e})"}
    train = pd.DataFrame([stats(p) for p in DATA.glob("*/*.csv")]).drop(columns="verdict")
    df = pd.DataFrame(rows).T
    ref = pd.DataFrame({"TRAIN 10%": train.quantile(0.1), "TRAIN median": train.median()}).T
    pd.set_option("display.width", 200)
    print(pd.concat([df, ref]).to_string(float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
