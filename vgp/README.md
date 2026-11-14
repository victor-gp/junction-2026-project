# vgp: DTW sign classifier (vocab 4: MOI, KIITOS, EI, OKEI)

Victor's prototype. Template matching with Dynamic Time Warping + k-nearest neighbours on the Rokoko glove CSVs in `../data/`.

```bash
cd vgp
uv sync                                   # numpy, pandas, scikit-learn, numba, matplotlib
uv run python scripts/eval.py             # cross-validation -> results/
uv run python scripts/train.py            # fit on all clips -> models/dtw_vocab4.npz
uv run python scripts/exp_round_shift.py  # centring / session-norm experiment
```

```python
from signrec.dtw import DTWKNN
m = DTWKNN.load("models/dtw_vocab4.npz")
m.predict(df)  # df: one clip in the training CSV format, 30 fps -> {"label", "nearest_distance", ...}
```

## Results (70 clips, 3 recording rounds)

| Split | Best accuracy |
|---|---|
| Random 5-fold CV (5 repeats) | 96–98% |
| Leave-one-recording-round-out | ~60% |

The gap means a new recording session looks very different from earlier ones (joint angle offsets of 20–50°, and differences in how signs are performed). Random CV overestimates live accuracy. Figures: `results/confusion_*.png`, `results/reject_distances.png`.
