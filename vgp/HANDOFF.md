# Handoff: live sign recognition (branch `fix-model-for-live`)

> **Superseded (2026-11-15 03:40): the demo is now the Studio clip demo. On the Windows Studio machine, follow [CLIP_DEMO_WINDOWS.md](CLIP_DEMO_WINDOWS.md).** The SDK live path below doesn't recognise signs reliably; this file is kept as history.

State as of 2026-11-15. Read this first, then `vgp/README.md` ("Live demo").

## Where things stand

The live recogniser runs end to end with both Smartgloves (Windows, Rokoko Device SDK). **But in the one live test so far, it recognised nothing**: 2 min, model trained only on Rokoko Studio exports, no words, never above 0.66 confidence, mostly Idle. Cause: the live SDK solver's finger features are about 2× further from the training data than Studio clips (mean |z| 1.42 vs 0.75; left index finger and right thumb joints up to 6 std off). It isn't a mirroring bug: flipping any axis makes the match worse.

Mitigations are in, but **they haven't been checked against labelled live input yet**. The real fix, training on clips recorded through the live path, is ready to run but not done.

## Run it

From `vgp/`, gloves connected, Rokoko Studio closed:

```powershell
..\venv\Scripts\python.exe scripts\live.py --adapt --log            # newest model
..\venv\Scripts\python.exe scripts\live.py --adapt --log --model models\live_cnn_prev.npz   # previous, A/B
..\venv\Scripts\python.exe scripts\live.py --replay ..\takes\live_logs\<stamp>_raw.csv --adapt --trace   # offline
```

`--adapt` needs 10 s of stream before it is active. `--log` writes `takes/live_logs/<stamp>_predictions.csv` (per window: all class probabilities, top class, word) and `<stamp>_raw.csv` (every solver frame; works with `--replay`).

## Next steps (in order)

1. **Record labelled live takes:** `..\venv\Scripts\python.exe scripts\record_prompted.py --reps 10` (70 takes × 4 s, ~8 min, random order, Idle included, saved as `data/<Label>/live_<session>_<n>.csv`). Untested with gloves (only `--dry-run`). Use the demo signer.
2. **Retrain on Modal** from the repo root (CLI in `venv`, logged in as profile `victor-gp`):
   `.\venv\Scripts\modal.exe run modal-gpu/run.py --workdir vgp --cmd "python scripts/train_live.py --eval --holdout 5"`
3. **Validate:** `python scripts/validate_live.py` (from `vgp/`). Live takes in the holdout are the first real live measurement. To check live alone: `--list` with a JSON of only `live_*` clips.
4. Then decide `--adapt` on/off and the decoder thresholds from measured live numbers, not from guesses.

## What was done on this branch

| Commit | Change |
|---|---|
| `035003b` | Bug fix: near-constant features (std ~1e-6 because Studio's skeleton is rigid) passed the mask due to float32 rounding, so tiny live offsets became huge z-scores and the model was stuck on "Thank you 1.00". Mask/stats now in float64 with a 0.01 floor (71 → 60 features), plus a guard in `LiveSignModel.preprocess`. `modal-gpu/run.py` no longer uploads `venv/`. |
| `22ec3d7` | Retrained with the fix (5 classes). |
| `9f6abed` | 7 classes (+ Help, World), second recording session added. `train_live.py --holdout N` (every Nth clip per class kept out) + `scripts/validate_live.py`. |
| `bb49cb4` | Robustness to new sessions: strong augmentation by default (offset ±2 std, gain ±50%, 30% feature dropout); `signrec.live.Adapter` / `live.py --adapt` (running 60 s mean/std alignment, causal); `live.py --log`; `scripts/record_prompted.py`; `--holdout-list`, `--aug-*`, `--feat-drop` for experiments. |
| `3ee57fd` | README fix. |

## Numbers

| Model | Same-session holdout (36 clips) | Train session 1 → test session 2 (45 clips) |
|---|---|---|
| `live_cnn_prev.npz` (mild augmentation) | 97% clip | 11%, 36% with alignment |
| `live_cnn.npz` (strong augmentation, current) | 89% clip, 94% window | 22%, **51% with `--adapt`** |

Live decoder on the holdout (current model, confidence 0.7 / votes 4): right word for 58% of clips, 3 wrong words (Idle → Help, No → World, Thank you → Hello). The other misses are clips under ~3.5 s (too few windows for 4 votes) or under-confident predictions. The previous model emitted no wrong words (75% right). Session lists: `results/live_holdout.json`, `results/session2_clips.json` (clips added after commit `19c1eed`).

## Not committed (local only)

- **`data-capture/config.py`**: local edit `CONFIDENCE 0.7 → 0.3`, `VOTES 4 → 2`. Not validated. 0.3 is close to what the Studio-trained model output on live Idle, so expect false words. Decide before committing.
- **`data/`**: 95 clips from the second session (incl. all of `Help/`, `World/`). The current model was trained on them, so commit them or nobody can reproduce it.
- **`takes/live_logs/20261115_010406_*`**: the first live test (unlabelled).
- **`streaming/`**: `SDK_STREAM_FORMAT.md` + `sdk_stream_samples.zip` for the teammate (raw driver stream sample; the solver sample is header only, no gloves streamed at the time).

## Gotchas

- `data/` sub-folders are the classes. A take outside `data/` (e.g. `takes/`) is never trained on, and an empty folder becomes a class that is never predicted.
- `main.py record` / the SDK reader can hang on exit. Check for leftover `rkk-hand-solver` / `rokoko-sdk` processes.
- On this machine, `torch` isn't installed locally. Train on Modal; inference needs only numpy + pandas (`venv`).
