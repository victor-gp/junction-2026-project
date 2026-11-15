"""All tunable parameters in one place.

Modules read these as `config.NAME` at call time, so they can also be
overridden from code before use, e.g. `config.WINDOW_S = 2.0`.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")          # Studio CSVs, one sub-folder per class

# ── Rokoko Device SDK ────────────────────────────────────────────────────────
SDK_BIN = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Rokoko Device SDK", "bin")
SOLVER_EXE = os.path.join(SDK_BIN, "rkk-hand-solver.exe")
DRIVER_EXE = os.path.join(SDK_BIN, "rokoko-sdk.exe")
DRIVER_ADDR = ("127.0.0.1", 12276)             # raw sensors (driver)
SOLVER_ADDR = ("127.0.0.1", 12277)             # solved hands (read by GloveStreamReader)
START_SOLVER = True                            # False if you run rkk-hand-solver yourself
CONNECT_TIMEOUT_S = 20.0

# ── Buffer ───────────────────────────────────────────────────────────────────
STREAM_RATE_HZ = 83.3                          # solver output rate (used to size the buffer)
BUFFER_SECONDS = 120.0

# ── Features ─────────────────────────────────────────────────────────────────
FEATURE_RATE_HZ = 30.0                         # model frame rate (Studio exports are 30 Hz)

# ── Windows and transcription ────────────────────────────────────────────────
IDLE_LABEL = "Idle"           # class meaning "no sign"; never emitted as a word
WINDOW_S = 3.0                # length of one model input; shortest training clip is 2.4 s
HOP_S = 0.25                  # how often the model runs live
TRAIN_HOP_S = 0.25            # step between training windows cut from a clip
EDGE_TRIM_S = 0.0             # cut from start and end of each training clip (resting hands)
MIN_FILL = 0.9                # min fraction of a full window that must be present live
CONFIDENCE = 0.3              # below this top probability, the window counts as Idle
VOTES = 2                     # consecutive windows that must agree before a word is emitted
