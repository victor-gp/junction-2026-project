# Spoken words for the clip demo

Generated with [Piper](https://github.com/OHF-Voice/piper1-gpl) by `scripts/make_audio.py` (16-bit mono WAV, 22050 Hz). Played by `signrec/voice.py`.

| File | Voice | Voice dataset licence |
|---|---|---|
| Hello.wav, World.wav | `en_US-hfc_female-medium` | Hi-Fi-CAPTAIN (NICT), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| No.wav, Okay.wav | `en_US-hfc_male-medium` | Hi-Fi-CAPTAIN (NICT), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| Help.wav, Thank you.wav | `en_US-amy-medium` | [Mycroft mimic3-voices](https://github.com/MycroftAI/mimic3-voices) |
| fi/*.wav (Moi, Apua, Ei, Okei, Kiitos, Maailma) | `fi_FI-harri-medium` (the only Finnish Piper voice) | Finnish single-speaker dataset (Kaggle), CC0 |

`audio/` is English (the default), `audio/fi/` is Finnish (`clip_demo.py --lang fi`).

Voice models: [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices). Non-commercial use only (hfc voices).
