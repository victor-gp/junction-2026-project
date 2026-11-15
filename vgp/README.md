# vgp: DTW sign classifier (vocab 4: MOI, KIITOS, EI, OKEI)

Victor's prototype. Template matching with Dynamic Time Warping + k-nearest neighbours on the Rokoko glove CSVs in `../data/`.

## Clip demo (Studio export → spoken word): the demo path

**Running it on the Windows Studio machine? Follow [CLIP_DEMO_WINDOWS.md](CLIP_DEMO_WINDOWS.md) step by step** (written for people and coding agents).

Sign one word in Rokoko Studio and export the take as CSV. `scripts/clip_demo.py` sees the new file, classifies the whole clip, prints the English word with its FinSL gloss and plays the word. It stays in the Studio data domain end to end, so there's no SDK-vs-Studio mismatch. Inference needs only numpy and pandas.

Model: `models/live_cnn.npz`, 7 classes, trained on all 182 Studio clips. A clip is cut into 3 s windows, the window probabilities are averaged, and the top class wins. If the top class is Idle, or its probability is below `--threshold` (default 0.5), it shows "not sure" and says nothing.

Held-out check (`results/live_cnn_holdout.npz`: same recipe, 36 clips kept out; the same split as `results/live_holdout.json`):

| threshold | right | wrong word | silent | Idle clips silent |
|---|---|---|---|---|
| 0.4 | 29/32 | 2 | 1 | 3/4 |
| **0.5** | **28/32** | **0** | 4 | **4/4** |
| 0.6 | 24/32 | 0 | 8 | 4/4 |

Grouped 5-fold CV on the 146 training clips: 79% clip accuracy. All of these clips come from the recorded sessions. **A new signer or session scores far lower** (11–51% before), so record about 5 clips per sign plus Idle from the demo signer, add them to `../data/<Label>/`, and retrain (below, about 3 min on Modal).

### Running it (from `vgp\`, Windows)

```powershell
py -m pip install numpy pandas
# once, with network: generate the voice clips into vgp\audio\ (stdlib only)
$env:ELEVENLABS_API_KEY = "..."      # never commit the key
py scripts\make_audio.py             # --voice <id> to pick another voice, --force to redo
py scripts\make_audio.py --play      # check the speakers
# demo
py scripts\clip_demo.py --watch "<Studio export folder>" --gui
```

- Run it on the Windows machine that has Studio. Python 3.10+ from python.org (keep "tcl/tk" ticked in the installer, which is the default). No torch, no extra audio packages.
- Don't end the `--watch` path with a backslash inside quotes (`"C:\x\"`): PowerShell then passes the quote as part of the path. Use `"C:\x"` or forward slashes.
- If Studio still holds a just-exported file locked, the read is retried for a few seconds before it's reported.
- `--gui` opens a big-text window (tkinter, included with python.org Python); without it, one line per clip in the terminal. Esc closes the window.
- Only CSVs that appear after start-up count; `--existing` also classifies the ones already there. Sub-folders are watched too. A re-export over the same file name is classified again.
- A file is read once it has stopped changing for 1 s (`--settle`), so Studio can finish writing it.
- Voice: `audio\<Label>.wav` if it exists (ElevenLabs), else Windows' built-in speech (offline), else silent. `--no-voice` mutes it.
- Offline check without gloves: `py scripts\clip_demo.py --files ..\data\No\Clip_44.csv ...`, or `--list results\live_holdout.json --model results\live_cnn_holdout.npz` for accuracy plus a threshold sweep.

Retrain after adding calibration clips (repo root, Modal):

```sh
modal run modal-gpu/run.py --workdir vgp --outputs models --cmd "python scripts/train_live.py"
```

## Live demo (gloves → words on screen)

A 1D-CNN on 3 s windows of the 114 live features from `../data-capture/features.py`, the same features for Studio CSVs and the live SDK stream. It runs every 0.25 s on the newest window, and a word appears after 4 agreeing windows. Classes = the sub-folders of `../data/`, `Idle` included.

- `models/live_cnn.npz`: the trained model. **Inference needs only numpy and pandas** (no torch).
- `scripts/live.py`: live runner and offline replay. A glove that drops out is frozen at its last pose, so recognition keeps going. Output is FinSL gloss + English, e.g. `EI · KIITOS → "No, thank you."`.
- `scripts/train_live.py`: training (torch; Modal or local CPU).
- `live_model.py`: the same model for oplew's runner: `python ../data-capture/main.py live --model live_model:load`.

Current model: 7 classes (Hello, Help, Idle, No, Okay, Thank you, World), 60 of the 114 features, trained on 146 Studio clips from two recording sessions with strong augmentation (per-feature offset ±2 std, gain ±50%, 30% feature dropout). On 36 held-out clips (`results/live_holdout.json`): **89% clip / 94% window**. Grouped CV: 80% clip. Strong augmentation costs some same-session accuracy but helps on a new session, which is what live input is. With session 1 only for training and session 2 for testing (`results/session2_clips.json`): 11% → 22% from augmentation, **51% with `--adapt`** (below). `models/live_cnn_prev.npz` is the previous model (97% same-session, 11%/36% cross-session).

**First live test (2 min, Studio-trained model): no words.** The live solver's finger features are about twice as far from the training data as Studio clips are (left index and right thumb up to 6 std off). Two mitigations are in place, and one real fix:

- `--adapt`: aligns each feature's running mean/std (last 60 s) with the training data (`signrec.live.Adapter`). Starts after 10 s, so rest your hands briefly at the start.
- the strong augmentation above.
- **the real fix: record training clips through the live path** with `scripts/record_prompted.py` (below), then retrain. **Record the demo signer.**

### Demo machine setup (Windows)

The live stream needs the Rokoko Device SDK, which `data-capture/rokoko_stream.py` expects on Windows.

1. **Python 3.10+** from <https://www.python.org/downloads/windows/>. Tick "Add python.exe to PATH" in the installer. Check in a new PowerShell: `py --version`.
2. **Get the code:** `git clone <repo url>`, then `cd junction-2026-project` and `git switch vgp/live-model` (or whichever branch Victor pushed).
3. **Packages** (inference only, no torch):

   ```powershell
   py -m pip install numpy pandas
   ```

4. **Rokoko Device SDK:** `rkk-hand-solver.exe` and `rokoko-sdk.exe` must be in `%LOCALAPPDATA%\Rokoko Device SDK\bin`. Check with:

   ```powershell
   dir "$env:LOCALAPPDATA\Rokoko Device SDK\bin"
   ```

   The SDK licence is tied to a Rokoko team; ask the Rokoko mentors. If the SDK lives elsewhere, change `SDK_BIN` in `data-capture/config.py`.
5. **Gloves:** connect both Smartgloves II. Close Rokoko Studio first. (Unverified: Studio may hold the gloves and keep the SDK driver from seeing them.)
6. **Run** from `vgp\`:

   ```powershell
   cd vgp
   py scripts\live.py --trace     # first: prints top class per window; check both gloves stream
   py scripts\live.py --adapt     # demo: one line per recognised word
   py scripts\live.py --adapt --log   # also log predictions + raw stream to ..\takes\live_logs\
   py scripts\live.py --adapt --model models\live_cnn_prev.npz   # previous model, for comparison
   ```

   The script starts the hand solver itself. It prints `(gloves connected: ...)` when fewer than two hands are seen. Stop with Ctrl+C.

#### Troubleshooting

- `hand solver not reachable`: the SDK isn't at the path above, the licence isn't active, or another program holds the port (12277).
- `model classes [...] != data/ folders [...]`: `data/` on this machine has different class folders than the model was trained on (e.g. new `Help`/`World` folders). Pull the retrained model, or pass `--no-class-check` to run with the model's own classes (`scripts/live.py` only).
- Words fire too often or too rarely: try `--confidence 0.8` / `--votes 6` (fewer false words, more lag), or `--confidence 0.6` / `--votes 3`.

### Recording more data

- **Prompted, through the live path (recommended):** `py scripts\record_prompted.py --reps 10` shows each sign with a countdown and saves 4 s takes to `data/<Label>/live_<session>_<n>.csv`, in random order, Idle included. `--dry-run` shows the plan. Then retrain.

- **Rokoko Studio CSV export** into `data/<Label>/Clip_N.csv`: works as is.
- **Through the live path** (demo machine, no Studio): `py ..\data-capture\main.py record --seconds 6 --out "..\data\Help\rec_01.csv"`. The loader detects this format (seconds, SDK quaternions, not mirrored), and it gives the same features as a Studio export of the same motion.
- **Continuous test takes** (whole demo script with pauses): record them **outside `data/`**, or they become a class. E.g. `takes\demo_1.csv`.

### Retraining (e.g. after adding `Help` and `World`)

New folders in `data/` become classes automatically. From the repo root:

```bash
modal run modal-gpu/run.py --workdir vgp --cmd "python scripts/train_live.py --eval"   # ~2 min on a T4
# or locally on CPU, from vgp/:  uv run --extra cnn python scripts/train_live.py --eval
```

This writes `vgp/models/live_cnn.npz` and `vgp/results/live_cv.json`. Commit both, then pull on the demo machine.

To check quality on clips the model never saw, add `--holdout 5`. Every 5th clip per class (in clip-number order, so every recording session is covered) is kept out of CV and the final fit, and listed in `results/live_holdout.json`. Then, from `vgp/`:

```bash
python scripts/validate_live.py    # clip / window accuracy + which words the live decoder would emit
```

Clips from the same recording session look alike, so this is still more optimistic than a new signer. Cross-session accuracy was ~20% before the second session was added to training.

### Tuning on recorded takes (no gloves needed)

```bash
uv run python scripts/live.py --replay ../takes/demo_1.csv            # words with timestamps
uv run python scripts/live.py --replay ../takes/demo_1.csv --trace    # + top 2 classes per window
uv run python scripts/live.py --replay ../takes/*.csv --confidence 0.8 --votes 5
```

Replay slides the same window over the file with the same decoder as live. Defaults come from `data-capture/config.py` (`CONFIDENCE` 0.7, `VOTES` 4, `HOP_S` 0.25, `WINDOW_S` 3.0). Gloss and phrase tables (`GLOSS`, `PHRASES`) are in `signrec/live.py`.

---

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
