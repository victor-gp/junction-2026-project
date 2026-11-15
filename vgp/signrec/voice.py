"""Spoken output for recognised signs: pre-generated WAV clips, offline fallback.

Clips are generated once (scripts/make_audio.py: Piper by default, or ElevenLabs) into
vgp/audio/<Label>.wav and committed, so the demo needs no TTS engine or network.
WAV plays on Windows with the stdlib `winsound`, no extra packages. (ElevenLabs
returns raw PCM, pcm_22050, which synthesize() wraps as WAV.)

Playback order per word: cached WAV -> system TTS (Windows SAPI via PowerShell,
`spd-say`/`espeak` on Linux) -> silent. Playback runs in a thread; it never blocks.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import wave
from pathlib import Path

AUDIO_DIR = Path(__file__).resolve().parents[1] / "audio"
API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=pcm_22050"
SAMPLE_RATE = 22050
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"      # example voice in the ElevenLabs API docs
DEFAULT_TTS_MODEL = "eleven_multilingual_v2"


def audio_path(label: str, audio_dir: Path = AUDIO_DIR) -> Path:
    return Path(audio_dir) / f"{label}.wav"


def voice_info(voice_id: str, api_key: str, timeout: float = 15.0) -> dict:
    """GET /v1/voices/<id>: name, labels (gender, accent, ...)."""
    req = urllib.request.Request(f"https://api.elevenlabs.io/v1/voices/{voice_id}",
                                 headers={"xi-api-key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"voice {voice_id}: ElevenLabs HTTP {e.code}: {e.read()[:300]!r}") from None


def synthesize(text: str, out: Path, api_key: str, voice_id: str = DEFAULT_VOICE,
               model_id: str = DEFAULT_TTS_MODEL, timeout: float = 30.0) -> None:
    """One ElevenLabs request -> 16-bit mono WAV at `out`."""
    req = urllib.request.Request(
        API_URL.format(voice_id=voice_id),
        data=json.dumps({"text": text, "model_id": model_id}).encode(),
        headers={"xi-api-key": api_key, "Content-Type": "application/json",
                 "Accept": "audio/pcm"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            pcm = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ElevenLabs HTTP {e.code}: {e.read()[:300]!r}") from None
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


# -- playback ------------------------------------------------------------------

def _play_wav(path: Path) -> bool:
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return True
    for cmd in (["paplay", str(path)], ["aplay", "-q", str(path)],
                ["afplay", str(path)], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=False)
            return True
    return False


def _system_tts(text: str) -> bool:
    if sys.platform == "win32":
        safe = text.replace("'", "''")
        ps = ("Add-Type -AssemblyName System.Speech; "
              f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return True
    for cmd in (["spd-say", "-w", text], ["espeak", text], ["say", text]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=False)
            return True
    return False


class Speaker:
    """speak(label, text): plays in a background thread; a new word waits for the last."""

    def __init__(self, audio_dir: Path = AUDIO_DIR, enabled: bool = True):
        self.audio_dir = Path(audio_dir)
        self.enabled = enabled
        self._lock = threading.Lock()

    def source(self, label: str) -> str:
        return "cached wav" if audio_path(label, self.audio_dir).exists() else "system tts"

    def _run(self, label: str, text: str) -> None:
        with self._lock:
            try:
                p = audio_path(label, self.audio_dir)
                if not (p.exists() and _play_wav(p)):
                    _system_tts(text)
            except Exception as e:      # audio must never kill the demo
                print(f"(audio failed: {e})", file=sys.stderr)

    def speak(self, label: str, text: str) -> None:
        if self.enabled:
            threading.Thread(target=self._run, args=(label, text), daemon=True).start()


def api_key_from_env() -> str | None:
    return os.environ.get("ELEVENLABS_API_KEY") or None
