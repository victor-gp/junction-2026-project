# Korvat - junction-2026-project

Thousands of deaf people in Finland use Finnish Sign Language, and there are only about 600 interpreters. We built a first step toward signers being understood without booking one.

Most sign-recognition research uses computer vision: a camera films the signer. That needs good light, the right angle and visible fingers, and a camera installed wherever the conversation happens, filming everyone in view. We use Rokoko Smartgloves II instead. They measure every finger joint directly, work anywhere, and belong to the signer: no venue has to install anything, and they record only hand movement, not faces or rooms.

This weekend we learned 6 Finnish Sign Language signs, recorded 182 examples ourselves, and trained a small neural network on hand-shape features. The recognised word appears on screen and is spoken aloud (ElevenLabs). On recordings it never saw in training, it gets 89% right, and at our confidence threshold it said no wrong words: when unsure, it stays quiet.

It's a proof of concept. Next: a larger vocabulary, more signers, live recognition straight from the gloves (we measured that the live SDK stream differs from studio recordings), and the other direction: smart glasses that show the hearing person's speech as a signing silhouette in the lens, so a conversation can flow both ways.

Challenge: Rokoko.

Technologies: Rokoko Smartgloves II, Rokoko Studio, Rokoko Device SDK, Python, NumPy, PyTorch, Modal (GPU training), ElevenLabs (speech).


## **PLEASE WATCH OUR DEMO VIDEOS** [HERE](https://www.youtube.com/playlist?list=PLCYQFsEnWO6I)
