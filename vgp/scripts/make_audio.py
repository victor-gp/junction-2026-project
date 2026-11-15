"""Generate the spoken word for each sign with ElevenLabs -> vgp/audio/<Label>.wav.

Run once, with network, before the demo (from vgp/; stdlib only):
    $env:ELEVENLABS_API_KEY = "..."          # PowerShell;  bash: export ELEVENLABS_API_KEY=...
    python scripts/make_audio.py [--voice VOICE_ID] [--force]

The key is read from the environment only; never commit it. Classes come from the
data/ folders (Idle is skipped); the English text from signrec.live.GLOSS.
Test playback afterwards:  python scripts/make_audio.py --play
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from signrec import voice  # noqa: E402
from signrec.live import GLOSS, config, realtime  # noqa: E402


def words() -> dict[str, str]:
    return {c: GLOSS.get(c, (None, c))[1] for c in realtime.classes() if c != config.IDLE_LABEL}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default=voice.DEFAULT_VOICE, help="ElevenLabs voice id")
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
    for label, text in words().items():
        out = voice.audio_path(label)
        if out.exists() and not a.force:
            print(f"{label:<10} exists  {out}")
            continue
        voice.synthesize(text, out, key, a.voice, a.tts_model)
        print(f"{label:<10} -> {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
