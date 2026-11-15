"""Generate the spoken word for each sign -> vgp/audio/<Label>.wav.

The clips are committed to the repo, so the demo machine needs no TTS engine, key or
network. Rerun this only to change the voices or add a word (from vgp/).

Three voices, two words each: two women, one man.

Default engine: Piper (free, offline neural TTS). Needs `pip install piper-tts` in
whatever Python runs this (not needed for the demo itself). Voice models (~60 MB
each) are downloaded once from Hugging Face into --model-dir.
    python scripts/make_audio.py [--force]

Alternative: ElevenLabs (needs an API key in the environment; never commit it):
    $env:ELEVENLABS_API_KEY = "..."          # bash: export ELEVENLABS_API_KEY=...
    python scripts/make_audio.py --engine elevenlabs --force
Each ElevenLabs voice's gender label is checked against the API before generating.

Test playback:  python scripts/make_audio.py --play
"""
import argparse
import sys
import time
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from signrec import voice  # noqa: E402
from signrec.live import GLOSS, config, realtime  # noqa: E402

# Piper voices (rhasspy/piper-voices on Hugging Face): (voice, gender, words)
PIPER_PLAN = [
    ("en_US-hfc_female-medium", "female", ["Hello", "Thank you"]),
    ("en_US-amy-medium", "female", ["Help", "World"]),
    ("en_US-hfc_male-medium", "male", ["No", "Okay"]),   # ryan clips short words
]
PIPER_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{locale}/{name}/{quality}/{voice}{ext}"
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "piper-voices"

# ElevenLabs premade voices: (voice id, expected gender label, words)
ELEVENLABS_PLAN = [
    ("EXAVITQu4vr4xnSDxMaL", "female", ["Hello", "Thank you"]),   # Sarah
    ("Xb7hH8MSUJpSbSDYk0k2", "female", ["Help", "World"]),        # Alice
    ("JBFqnCBsd6RMkjVDRZzb", "male", ["No", "Okay"]),             # George
]


def words() -> dict[str, str]:
    return {c: GLOSS.get(c, (None, c))[1] for c in realtime.classes() if c != config.IDLE_LABEL}


def check_plan(plan, todo) -> dict[str, str]:
    by_label = {label: v for v, _, labels in plan for label in labels}
    missing = sorted(set(todo) - set(by_label))
    if missing:
        sys.exit(f"no voice assigned for {missing}; edit the plan in make_audio.py")
    return by_label


# -- Piper ----------------------------------------------------------------------------

def piper_model(name: str, model_dir: Path) -> Path:
    """Download <name>.onnx (+ .onnx.json) once; return the .onnx path."""
    locale, rest = name.split("-", 1)
    speaker, quality = rest.rsplit("-", 1)
    onnx = model_dir / f"{name}.onnx"
    for ext in (".onnx.json", ".onnx"):
        dest = model_dir / f"{name}{ext}"
        if dest.exists():
            continue
        url = PIPER_URL.format(lang=locale.split("_")[0], locale=locale, name=speaker,
                               quality=quality, voice=name, ext=ext)
        print(f"downloading {url}", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    return onnx


def run_piper(todo, a) -> None:
    try:
        from piper import PiperVoice
    except ImportError:
        sys.exit("Piper is not installed: pip install piper-tts")
    by_label = check_plan(PIPER_PLAN, todo)
    for name, gender, labels in PIPER_PLAN:
        print(f"voice {name:<26} {gender:<6} -> {', '.join(labels)}")
    loaded: dict[str, object] = {}
    for label, text in todo.items():
        out = voice.audio_path(label)
        if out.exists() and not a.force:
            print(f"{label:<10} exists  {out}")
            continue
        name = by_label[label]
        if name not in loaded:
            loaded[name] = PiperVoice.load(str(piper_model(name, Path(a.model_dir))))
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as wf:
            loaded[name].synthesize_wav(text, wf)
        print(f"{label:<10} -> {out} ({out.stat().st_size // 1024} KB, {name})")


# -- ElevenLabs -------------------------------------------------------------------------

def run_elevenlabs(todo, a) -> None:
    key = voice.api_key_from_env()
    if not key:
        sys.exit("set ELEVENLABS_API_KEY first")
    by_label = check_plan(ELEVENLABS_PLAN, todo)
    for vid, gender, labels in ELEVENLABS_PLAN:
        info = voice.voice_info(vid, key)
        got = (info.get("labels") or {}).get("gender", "?")
        print(f"voice {info.get('name', '?'):<10} {vid}  gender={got}  -> {', '.join(labels)}")
        if got != gender:
            sys.exit(f"voice {vid} is '{got}', expected '{gender}'; edit ELEVENLABS_PLAN")
    for label, text in todo.items():
        out = voice.audio_path(label)
        if out.exists() and not a.force:
            print(f"{label:<10} exists  {out}")
            continue
        voice.synthesize(text, out, key, by_label[label], a.tts_model)
        print(f"{label:<10} -> {out} ({out.stat().st_size // 1024} KB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["piper", "elevenlabs"], default="piper")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Piper voice model cache")
    ap.add_argument("--tts-model", default=voice.DEFAULT_TTS_MODEL, help="ElevenLabs model id")
    ap.add_argument("--force", action="store_true", help="regenerate existing clips")
    ap.add_argument("--play", action="store_true", help="only play every word (checks audio output)")
    a = ap.parse_args()

    if a.play:
        sp = voice.Speaker()
        for label, text in words().items():
            print(f"{label:<10} ({sp.source(label)})", flush=True)
            sp._run(label, text)
            time.sleep(0.3)
        return

    (run_piper if a.engine == "piper" else run_elevenlabs)(words(), a)


if __name__ == "__main__":
    main()
