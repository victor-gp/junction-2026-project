"""Generate the spoken word for each sign with ElevenLabs -> vgp/audio/<Label>.wav.

The clips are committed to the repo, so the demo machine needs neither a key nor the
network. Rerun this only to change the voices or add a word (from vgp/; stdlib only):
    $env:ELEVENLABS_API_KEY = "..."          # PowerShell;  bash: export ELEVENLABS_API_KEY=...
    python scripts/make_audio.py [--force]

Three voices, two words each (VOICE_PLAN): two women, one man. Each voice's gender
label is checked against the API before generating. --voice <id> uses one voice for
every word instead. The key is read from the environment only; never commit it.
Test playback:  python scripts/make_audio.py --play
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from signrec import voice  # noqa: E402
from signrec.live import GLOSS, config, realtime  # noqa: E402

# ElevenLabs premade voices: (voice id, expected gender label, words)
VOICE_PLAN = [
    ("EXAVITQu4vr4xnSDxMaL", "female", ["Hello", "Thank you"]),   # Sarah
    ("Xb7hH8MSUJpSbSDYk0k2", "female", ["Help", "World"]),        # Alice
    ("JBFqnCBsd6RMkjVDRZzb", "male", ["No", "Okay"]),             # George
]


def words() -> dict[str, str]:
    return {c: GLOSS.get(c, (None, c))[1] for c in realtime.classes() if c != config.IDLE_LABEL}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default=None, help="one ElevenLabs voice id for every word")
    ap.add_argument("--tts-model", default=voice.DEFAULT_TTS_MODEL)
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

    key = voice.api_key_from_env()
    if not key:
        sys.exit("set ELEVENLABS_API_KEY first")

    todo = words()
    if a.voice:
        plan = {label: a.voice for label in todo}
    else:
        plan = {}
        for vid, gender, labels in VOICE_PLAN:
            info = voice.voice_info(vid, key)
            got = (info.get("labels") or {}).get("gender", "?")
            print(f"voice {info.get('name', '?'):<10} {vid}  gender={got}  -> {', '.join(labels)}")
            if got != gender:
                sys.exit(f"voice {vid} is '{got}', expected '{gender}'; change VOICE_PLAN")
            plan.update({label: vid for label in labels})
        missing = sorted(set(todo) - set(plan))
        if missing:
            sys.exit(f"no voice in VOICE_PLAN for {missing}")

    for label, text in todo.items():
        out = voice.audio_path(label)
        if out.exists() and not a.force:
            print(f"{label:<10} exists  {out}")
            continue
        voice.synthesize(text, out, key, plan[label], a.tts_model)
        print(f"{label:<10} -> {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
