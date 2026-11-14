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

## 1D-CNN (needs the `cnn` extra: PyTorch)

Same features, splits and `predict()` interface as DTW. Training augments each clip on the fly: crop, time warp, noise, per-channel gain and offset, and freezing one hand to simulate a glove that stops emitting.

```bash
# on a Modal GPU (see modal_gpu.py); results/ and models/ are copied back
modal run modal_gpu.py --cmd "python scripts/eval_cnn.py && python scripts/train_cnn.py"
# variants
modal run modal_gpu.py --cmd "python scripts/eval_cnn.py --features angles+handpos+vel --tag vel"
modal run modal_gpu.py --cmd "python scripts/eval_cnn.py --no-aug --tag noaug"
# local code check only (2 epochs, chance-level accuracy expected)
uv run --extra cnn python scripts/eval_cnn.py --smoke --tag smoke
```

```python
from signrec.nn import CNNClassifier
m = CNNClassifier.load("models/cnn_vocab4.pt")
m.predict(df)  # {"label", "confidence", "probs", "rejected", ...}
```
