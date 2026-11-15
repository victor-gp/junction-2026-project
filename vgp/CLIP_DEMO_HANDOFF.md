# Handoff: clip demo, state as of 2026-11-15 08:00

Read this first. The step-by-step setup is in [CLIP_DEMO_WINDOWS.md](CLIP_DEMO_WINDOWS.md); this file says what works, what doesn't, and what to do next.

## Summary

- **The demo pipeline works.** A Studio CSV dropped into the watched folder is classified in under 0.2 s, shown in the big window and spoken. It was checked on the Windows Studio machine with real Studio exports.
- **New recordings are blocked by Rokoko Studio, not by our code.** Since about 04:30 every new take is saved without glove motion, so its export is one frozen pose and the demo answers "not sure" or a wrong word. The gloves move in Studio's live view, but replay and export are frozen.
- **What works for the demo today:** replaying recorded clips that the model never trained on (section 2). This gives 28 of 32 words right, none wrong, and all Idle clips silent.

## 1. What was verified

| Check | Result |
|---|---|
| Offline: hold-out model on 36 held-out clips (`--list results\live_holdout.json --model results\live_cnn_holdout.npz`) | 89% argmax. At threshold 0.5: **28 right, 0 wrong, 4 silent, Idle 4/4 silent** |
| Watcher with a copied training clip | `>>> OKAY (OKEI) 0.92` in 40 ms |
| Real Studio export of an **old** take (Clip 164, re-exported today) | Identical to `data/Okay/Clip_164.csv` (max difference 0.0), and the demo says OKAY 0.92. **Studio export is fine.** |
| Real Studio exports of **new** takes (Clip 202–208, fresh-scene "Clip") | All frozen (see section 3). The demo gives "not sure", or once a wrong "Hello" for a signed "No" |
| `scripts\replay_clips.py`, one held-out clip per word, into a running demo | Hello, Help, No, Okay, Thank you spoken correctly; Idle and one World stayed silent; no wrong words |

Machine facts:
- Studio saves exports to `%USERPROFILE%\AppData\LocalLow\Rokoko Electronics\Rokoko Studio\Exports`.
- Studio does **not** auto-export: each take needs Export → CSV.
- Studio's CSV settings (positions, 30 fps) match the training data.

## 2. Running the demo that works (replay)

From `vgp\`, in two PowerShell windows:

```powershell
# window 1: the demo, with the model that did NOT train on the hold-out clips
py scripts\clip_demo.py --watch "$env:USERPROFILE\AppData\LocalLow\Rokoko Electronics\Rokoko Studio\Exports" --gui --model results\live_cnn_holdout.npz

# window 2: feed it clips, as if Studio had just exported them
py scripts\replay_clips.py --watch "$env:USERPROFILE\AppData\LocalLow\Rokoko Electronics\Rokoko Studio\Exports" --per-word 1 --interval 5
```

- The demo window comes to the front by itself on each recognised word, without taking keyboard focus. Idle and "not sure" results never raise it, flash it or speak. Other display options:
  - `--fullscreen` (or F11) for a projector
  - `--on-top` to keep it above everything
  - `--no-raise` to switch the auto-raise off
- `replay_clips.py` options:
  - default: all 36 hold-out clips
  - `--words Hello Okay "Thank you"`: only these words
  - `--per-word N`: at most N clips per word
  - `--shuffle`: random order
  - `--files <csv> ...`: specific files
- Copies are named `replay_<time>_<Label>_<clip>.csv`. You can delete them from the folder afterwards.
- **Be honest about what's shown.** These are recorded takes from the training session, replayed, not live signing. Use the hold-out model with hold-out clips, so each clip is new to the model. The default `models\live_cnn.npz` trained on all 182 clips, so replaying with it proves nothing.
- Hold-out clips that stay silent at 0.5: `No/Clip_201`, `Thank you/Clip_16`, `Thank you/Clip_39`, `World/Clip_103`. Skip them with `--files` if you need every word spoken.
- Voice: the six words are pre-generated WAVs in `audio\` (Piper, shipped in the repo). The demo prints `voice: cached wav`; check the speakers with `py scripts\make_audio.py --play` (runbook step 4).

## 3. The blocker: Studio saves takes without glove data

Symptom: the gloves move live in Studio, but replay of the take and its CSV export are frozen. Check any export with:

```powershell
py scripts\check_motion.py "$env:USERPROFILE\AppData\LocalLow\Rokoko Electronics\Rokoko Studio\Exports"
```

| | Wrist travel | Finger bend range | Verdict |
|---|---|---|---|
| Training clips (median) | ~0.19 / 0.29 m | ~15–18° | moving |
| Clip 164 re-exported today | 0.02 / 0.39 m | 1.4 / 60° | moving |
| Clip 202–208, "Clip" (new takes) | 0.00001–0.0001 m | 0.01–0.07° | **FROZEN** |

(No training clip is flagged FROZEN, so the check doesn't give false alarms.)

### What we found

- **Studio's saved take files tell the story.** These live under `Production\<team>\<project>\<scene>\<take id>.data`.
  - Good takes (Clip 101–201) are **186–343 KB** and contain glove sensor curves (Quaternion, Gyro, Position, EMF).
  - Every take since 04:38 is **~12.8 KB whatever its length**: calibration and setup data only, no motion keys.
- **During recording the data is there.** The log (`Production\logs\studio.log`) shows about 420 frames captured per glove per take. It never reaches the saved take.
- **Ruled out:**
  - **Export settings:** an old take re-exports perfectly with today's settings.
  - **The leftover glove entries in the team scene:** a fresh scene with only our two gloves behaves the same.
  - **A Studio restart:** after restarting, Clip 208 is still 12.7 KB and frozen.
  - **The demo or model code:** real good exports classify correctly.
- **What changed before it broke:**
  - The good takes used glove IDs like R6FI, CE5F, WYXI and GT19.
  - Around 04:00–04:27 the gloves were set up again over USB and moved to Wi-Fi. They now show as `Smartgloves (0747) Left` and `Smartgloves (7) Right`.
  - The right glove's 1-character ID and both gloves reporting the same serial number look suspicious.
  - The log also shows cloud-sync errors ("disposed object") and a device-service "bad function call" at 02:40.

### My estimate of the cause (judgement, not measured)

| Cause | Odds |
|---|---|
| Glove hardware faulty | ~10%. The live view is smooth and the recorder captures frames. |
| Glove setup (pairing, IDs) | ~40%. It broke right after the gloves were set up again. |
| Rokoko Studio saving or sync | ~50%. Data is captured live but missing from the saved take. |

## 4. Next steps, in order

1. **Ask whoever recorded Clip 101–201** (data commits by `oplew`): what glove setup, scene and actor they used, and whether R6FI, CE5F and the others are these same physical gloves. Restoring that setup is the most likely fix.
2. **Re-pair the gloves:** remove both from Studio's device list and re-add them over USB with Rokoko's Smartgloves II pairing steps. Then record 4 s and **play it back in Studio** before exporting. If playback moves, export it and run `check_motion.py`.
3. **Fresh local project** (not a new scene inside the team project), offline if possible. This rules the team cloud sync in or out.
4. **SDK test of the gloves without Studio:** close Studio, then run `py data-capture\main.py record --seconds 6 --out test.csv` from the repo root and check it moves. This needs the Rokoko Device SDK in `%LOCALAPPDATA%\Rokoko Device SDK\bin`, which **isn't installed on this machine**. Copy it from the machine that made `takes/live_logs/`, or ask the Rokoko mentors.
5. **Rokoko support:** "Studio 2.6.2, Smartgloves II firmware 3.4.0-r: live view moves, recorded takes contain no glove curves (take .data ~13 KB)".
6. **Once recordings move again:** do the calibration in runbook step 7 (about 5 takes per word from the demo signer, then retrain). It's needed because new-signer accuracy was 11–51% in earlier tests. Retraining needs PyTorch (not installed here) or Modal (not set up here).

## 5. Files

| File | What |
|---|---|
| `scripts/clip_demo.py` | The demo: watches the export folder, classifies, shows and speaks |
| `scripts/replay_clips.py` | **New.** Feeds recorded clips into the watched folder (section 2) |
| `scripts/check_motion.py` | **New.** Flags frozen exports by comparing with the training data (section 3) |
| `results/live_cnn_holdout.npz`, `results/live_holdout.json` | Model without the 36 hold-out clips, and the list of those clips |
| `models/live_cnn.npz` | Model trained on all 182 clips: use it for real new recordings, not for replay |
| `CLIP_DEMO_WINDOWS.md` | Setup runbook |

Small known issue: in `ExportWatcher.retry` (`scripts/clip_demo.py`), the failure count per file is never reset. A file name that failed 4 reads earlier gets only one retry after a later re-export. It's harmless for the demo.
