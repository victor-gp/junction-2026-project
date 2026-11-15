# Runbook: clip demo on the Windows Studio machine

> **Status 2026-11-15:** read [CLIP_DEMO_HANDOFF.md](CLIP_DEMO_HANDOFF.md) first. New Studio takes are currently saved without glove motion; the demo runs on replayed hold-out clips until that is fixed.

For a person or a coding agent on the Windows machine that has Rokoko Studio. Follow the steps in order. Each step says what you should see; stop and fix it before going on.

**This is the demo path.** Ignore the SDK "Live demo" instructions in `README.md` and `HANDOFF.md`. The SDK live stream doesn't recognise signs reliably (its hand data differs from the Studio training data).

What the demo does: someone signs one word in Studio and exports the take as CSV. `scripts/clip_demo.py` notices the new file, classifies the clip, shows the word in a big window and says it out loud. Seven classes: Hello, Help, Idle, No, Okay, Thank you, World. Idle is resting hands and is never said.

All commands are PowerShell, run from the `vgp\` folder unless stated.

## 1. Get the code

```powershell
git clone <repo url>          # skip if the repo is already there
cd junction-2026-project
git fetch
git switch vgp/clip-demo
cd vgp
```

Expected: `scripts\clip_demo.py`, `models\live_cnn.npz` and `CLIP_DEMO_WINDOWS.md` exist.

## 2. Python and packages

Python 3.10 or newer from <https://www.python.org/downloads/windows/>. In the installer, tick "Add python.exe to PATH" and keep the default "tcl/tk" option (the window needs it).

```powershell
py --version
py -m pip install numpy pandas
py -c "import numpy, pandas, tkinter; print('ok')"
```

Expected: `ok`. No torch is needed on this machine, and no audio packages (sound uses Windows' built-in `winsound`).

## 3. Check the model offline (no gloves)

```powershell
py scripts\clip_demo.py --list results\live_holdout.json --model results\live_cnn_holdout.npz --quiet
```

Expected: `36 labelled clips, argmax clip accuracy 89%`, and at threshold 0.5: `28 right, 0 wrong, 4/4 Idle silent`. If you see `model classes [...] != data/ folders [...]`, the `..\data\` folders on this machine differ from the seven above; pull the branch again.

## 4. Voice (already in the repo; just check the speakers)

The six spoken words are pre-generated WAV files in `audio\` (Piper, a free offline voice model; three voices: two women, one man). Nothing to install, no key, no network.

```powershell
py scripts\make_audio.py --play       # plays every word through the speakers
```

Expected: six words (Hello, Help, No, Okay, Thank you, World), each line tagged `(cached wav)`. If a line says `(system tts)`, that WAV is missing: check that `audio\<Label>.wav` exists (pull the branch again). The demo then falls back to Windows' built-in speech.

Don't regenerate the clips on the demo machine. If they must change, see the top of `scripts\make_audio.py` (needs `pip install piper-tts`, or an ElevenLabs key with `--engine elevenlabs`).

## 5. Studio export folder

Ask the human two things (these aren't known yet):

1. **Where does Studio save CSV exports?** That folder is `--watch`.
2. **Does Studio write the CSV automatically when a recording stops, or does each take need an export click?** If it's manual, the demo flow is "record, export as CSV, wait for the word".

The export must be **CSV**, the same kind as the training clips in `..\data\<Label>\Clip_N.csv` (columns like `LeftDigit2Proximal_position_x`, timestamps in ms). FBX or other formats are ignored.

## 6. Run the demo

```powershell
py scripts\clip_demo.py --watch "C:\path\to\Studio exports" --gui
```

- Don't end the path with `\` inside quotes (`"C:\x\"` breaks in PowerShell). Forward slashes also work.
- Expected terminal output: the model line, `voice: elevenlabs` (or `system tts`), `watching <folder>`. A dark window shows "sign, then export".
- Record one sign in Studio and export it. Within about 2 s the window shows the English word, the FinSL gloss and a confidence, the same appears as a popup over all windows (bottom of the screen, fades after 4 s, doesn't take focus from Studio), and the word is spoken. If unsure, it shows "not sure" and stays silent.
- Only files that appear after start-up count; add `--existing` to also classify files already there. Sub-folders are watched. Re-exporting to the same file name classifies it again.
- `--overlay-only` shows only the popups over Studio (main window minimised). `--popup-seconds`, `--popup-pos` (`bottom`, `top`, `center`, `top-right`, `bottom-right`) and `--no-popup` change the popup.
- Without `--gui`: one line per clip in the terminal. `--no-voice` mutes it. Esc or closing the window quits.

Quick test without gloves: start the demo, then copy a training clip into the watched folder:

```powershell
Copy-Item "..\data\Okay\Clip_164.csv" "C:\path\to\Studio exports\test.csv"
```

Expected: `>>> OKAY (OKEI) 0.92`.

## 7. Calibrate on the demo signer (important)

The model is accurate on people and sessions it was trained on, and much worse on a new signer or day (11–51% in earlier tests). Right before the demo:

1. In Studio, record the demo signer: about 5 takes of each sign, plus 5 of resting hands (Idle), 3 to 4 s each. Calibrate the gloves in Studio first (Straight Pose) as usual.
2. Export each take as CSV into the matching class folder: `..\data\Hello\`, `..\data\Help\`, `..\data\Idle\`, `..\data\No\`, `..\data\Okay\`, `..\data\Thank you\`, `..\data\World\`. Give them new names, e.g. `Calib_1.csv`, `Calib_2.csv`. **Never rename, move or delete existing files in `data\`**; the folder names are the labels.
3. Retrain on Modal (from the **repo root**, not `vgp\`; about 3 minutes):

   ```powershell
   py -m pip install modal       # once
   py -m modal setup             # once; logs in through the browser
   py -m modal run modal-gpu/run.py --workdir vgp --outputs models --cmd "python scripts/train_live.py"
   ```

   Expected at the end: `saved /repo/vgp/models/live_cnn.npz classes=[...7...]` and `Copied back into ...\vgp: ['models']`. Modal needs an account with a payment method (see `modal-gpu/README.md`). If the human already uses Modal on another machine, it can run there instead after copying the new CSVs into its `data\`.

4. Restart `clip_demo.py`. Test with a couple of fresh takes from the demo signer.

Ask the human before committing or pushing new clips or models; the repo is shared.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Nothing happens after an export | Wrong `--watch` folder, or the export isn't CSV. Check where the file actually landed. |
| `could not read (None of [...] are in the [columns])` | Not a Studio CSV with finger positions (wrong export type, or gloves not in the take). |
| Always "not sure" | New signer: do step 7. To see what it would say, run with `--threshold 0.3` (more wrong words). |
| Wrong words | Raise `--threshold` to 0.6 (more "not sure"). Recalibrate (step 7). |
| No sound | `py scripts\make_audio.py --play`. Check Windows output device and volume. |
| Window text tiny or huge | Resize the window; fonts are fixed in `run_gui` in `scripts/clip_demo.py`. |
| `model classes ... != data/ folders` | `data\` has an extra or missing class folder. Every class folder needs CSVs; retrain after changing folders. |
