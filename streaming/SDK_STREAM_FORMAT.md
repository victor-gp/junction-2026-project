# Rokoko Smartgloves II: live SDK stream format

What the Rokoko Device SDK sends to our program, with real samples. Everything comes from local TCP streams on the demo machine. Rokoko Studio is not involved.

```text
gloves â”€â”€â”€â”€â”€â”€> rokoko-sdk.exe (driver)  â”€â”€TCP 127.0.0.1:12276â”€â”€> raw sensors        (stream 1)
                       â”‚
                       â””â”€â”€> rkk-hand-solver.exe --emit both â”€â”€TCP 127.0.0.1:12277â”€â”€> solved hands (stream 2)
                                                                                         â”‚
                                         data-capture/rokoko_stream.py (GloveStreamReader) â”˜
                                         â†’ FrameBuffer â†’ features â†’ model
```

Our live recogniser reads only **stream 2** (solved hands). Stream 1 is documented because that's the one we have real samples of.

## Files in this package

| File | What it is |
|---|---|
| `SDK_STREAM_FORMAT.md` | this document |
| `column_check.csv` | **Stream 1, real data**: 2.2 s of the raw driver stream, one glove (366 rows Ã— 159 columns) |
| `column_check_after_hand_solver.csv` | **Stream 2, header only**: the exact column layout of solved hands as our program stores it (281 columns, 0 data rows) |
| `studio_export_example_Hello_Clip_10.csv` | for comparison: one Rokoko Studio CSV export from our training set (`data/Hello/`) |

> **Missing:** we don't have solved-hand data rows yet. Both attempts to record stream 2 returned 0 frames because no gloves were streaming at the time. To make one with both gloves connected and Rokoko Studio closed, run from the repo root:
> `python data-capture/main.py record --seconds 10 --out streaming/solver_sample.csv`

---

## Transport (both streams): RGMP v2 over TCP

Taken from `data-capture/rokoko_stream.py`, which follows the SDK's `examples/python/rgmp_stream.py`. Little-endian throughout.

Every message:

```text
uint32 msg_prefix   # 1 = DEFINITION, 2 = DATA, 3 = DISCONNECT
uint32 msg_len      # payload length in bytes
byte[msg_len] payload
```

- **DEFINITION (1):** UTF-8 JSON describing one device. It contains `device_id`, `device_type` (e.g. `"solved_hand"`), `device_info.hand` (`"left"`/`"right"`) and `groups[]`. Each group lists `streams[]`, and each stream has `target_frame`, `measure_type` (`TRANSFORM`, `ORIENTATION`, â€¦), `reference_frame` (e.g. `LTP_NED`) and `data_type` (e.g. `FLOAT[7]`). The order of the streams gives the byte layout of that group's DATA payload.
- **DATA (2):** header `uint32 device_id, uint32 group_id, uint64 timestamp_us`, then the group's streams packed back to back in definition order. Data types are `INT32/UINT32/INT64/UINT64/FLOAT/DOUBLE[n[,m]]`.
- **DISCONNECT (3):** `uint32 device_id`.

One sample from one device can arrive as several DATA messages (one per group) that share a `timestamp_us`.

---

## Stream 1: raw driver (`column_check.csv`)

**Shape:** 366 rows Ã— 159 columns (unnamed row index, `Timestamp`, 157 sensor values), one glove.

- **Timestamp:** seconds on the device clock (`2001.92 â€¦ 2004.10`), not wall time.
- **Rate:** 183 distinct timestamps in 2.18 s = **83.4 Hz**.
- **2 rows per timestamp:** one row was written per incoming group (`default`, then `emf_only`), with the other group's previous values carried over. Row 0 has the `emf_only` columns empty (NaN) because that group hadn't arrived yet. **Use one row per timestamp (e.g. the last).**

Column naming: `<group>/<sensor>/<MEASURE>_<i>`.

### Group `default`: fused sensor values (116 columns)

For each of `thumb_finger`, `index_finger`, `middle_finger`, `ring_finger`, `little_finger`, `arm` (15 columns each):

| Columns | n | Meaning | Example (index finger) |
|---|---|---|---|
| `<s>_sensor/TRANSFORM_0..2` | 3 | position, metres | `0.0331, -0.0712, 0.0743` |
| `<s>_sensor/TRANSFORM_3..6` | 4 | orientation, unit quaternion (norm = 1.000) | `-0.6193, -0.3683, 0.0112, 0.6933` |
| `<s>_imu/ANGULAR_VELOCITY_0..2` | 3 | gyroscope (units not documented; probably rad/s) | `-0.3597, 0.4385, 0.3104` |
| `<s>_imu/PROPER_ACCELERATION_0..2` | 3 | accelerometer, in g (the norm averages 1.00 at rest) | `0.1426, 0.4713, -0.7132` |
| `<s>_sensor/STATUS_FLAGS` | 1 | status bit field, always `1` in this capture | `1` |

Hub (back of the hand), 11 columns:

| Columns | n | Meaning | Example |
|---|---|---|---|
| `hub/ORIENTATION_0..3` | 4 | unit quaternion | `-0.1681, 0.2836, 0.6173, 0.7143` |
| `hub_imu/ANGULAR_VELOCITY_0..2` | 3 | gyroscope | |
| `hub_imu/PROPER_ACCELERATION_0..2` | 3 | accelerometer, g | |
| `hub/STATUS_FLAGS` | 1 | always `2049` (bits 0 and 11) | `2049` |

The `default` group also contains a **second `arm_sensor` block** (`TRANSFORM_0..6`, `CUSTOM`, `STATUS_FLAGS`). Pandas renamed its duplicate column names to `TRANSFORM_0.1 â€¦ TRANSFORM_6.1`, `STATUS_FLAGS.1`. Its values differ from the first arm block, so it is probably a different reference frame. `default/arm_sensor/CUSTOM` is always 0.

### Group `emf_only`: electromagnetic-field positions only (54 columns)

For each of the 5 fingers and `arm` (9 columns each):

| Columns | n | Meaning | Example (index finger) |
|---|---|---|---|
| `<s>_sensor/TRANSFORM_0..6` | 7 | position (m) + quaternion, from EMF tracking alone (no IMU fusion); close to the `default` values | `0.0326, -0.0723, 0.0743, -0.6190, -0.3703, 0.0037, 0.6926` |
| `<s>_sensor/CUSTOM` | 1 | undocumented scalar. Mean per finger: thumb 4.3, index 5.8, middle 7.2, ring 7.9, little 10.5, arm 1.1 (maybe EMF signal quality or distance) | `3.97` |
| `<s>_sensor/STATUS_FLAGS` | 1 | always `0` | `0` |

The quaternion component order (`x,y,z,w` vs `w,x,y,z`) is not stated in the capture. Stream 2 uses `x,y,z,w`.

---

## Stream 2: solved hands (what the recogniser uses)

From the SDK, `rkk-hand-solver --emit both` sends one `solved_hand` device per connected glove. `GloveStreamReader` keeps only:

- `TRANSFORM` streams with `reference_frame == "LTP_NED"`, as **world position of the bone (metres; the hand bone itself is at 0)**
- `ORIENTATION` streams, as **rotation relative to the parent bone (quaternion x, y, z, w)**

It merges left and right into **one row per solved frame** and pushes it into a ring buffer at about **83.3 Hz**. A hand that isn't connected is all NaN.

**Shape of one row:** `Timestamp` (seconds, `timestamp_us / 1e6`) + **280 floats** = 2 hands Ã— 20 joints Ã— 7 (`position_x,y,z` + `rotation_x,y,z,w`). Exact header: `column_check_after_hand_solver.csv`. The leading unnamed column there is just the pandas row index.

Column order: both hands first, then all left digits, then all right digits:

```text
Timestamp,
LeftHand_position_{x,y,z},  LeftWrist_rotation_{x,y,z,w},
RightHand_position_{x,y,z}, RightWrist_rotation_{x,y,z,w},
Left<Bone>_position_{x,y,z}, Left<Joint>_rotation_{x,y,z,w}     Ã— 19 bones
Right<Bone>_position_{x,y,z}, Right<Joint>_rotation_{x,y,z,w}   Ã— 19 bones
```

Solver frame â†’ our column names (same names as Rokoko Studio CSV exports):

| Solver `target_frame` (`left_`/`right_` prefix) | Position column `<Side><Bone>_position_*` | Rotation column `<Side><Joint>_rotation_*` |
|---|---|---|
| `hand` | `Hand` | `Wrist` |
| `finger1_metacarpal` | `Digit1MetaCarpal` | `Digit1Carpometacarpal` |
| `finger1_proximal` | `Digit1ProximalPhalangx` | `Digit1Metacarpophalangeal` |
| `finger1_distal` | `Digit1DistalPhalanx` | `Digit1Interphalangeal` |
| `finger{2..5}_metacarpal` | `Digit{d}MetaCarpal` | `Digit{d}Carpometacarpal` |
| `finger{2..5}_proximal` | `Digit{d}ProximalPhalangx` | `Digit{d}Metacarpophalangeal` |
| `finger{2..5}_medial` | `Digit{d}IntermediatePhalanx` | `Digit{d}ProximalInterphalangeal` |
| `finger{2..5}_distal` | `Digit{d}DistalPhalanx` | `Digit{d}DistalInterphalangeal` |

(`Phalangx` is Rokoko's own spelling.) Thumb = Digit1, with 3 bones. Fingers 2â€“5 have 4 bones each. 1 + 3 + 4Ã—4 = 20 joints per hand.

### Differences from the Studio CSV exports (our training data)

| | Live solver (stream 2) | Studio export (`data/<Label>/*.csv`) |
|---|---|---|
| Timestamp | seconds (device clock) | milliseconds from 0 |
| Rate | ~83.3 Hz | 30 Hz (33 ms steps) |
| Positions | 120 columns, **same names** | 120 columns, **same names** |
| Rotations | 160 columns, quaternion `_rotation_{x,y,z,w}` | 120 columns, joint angles in **degrees** (`_flexion`, `_adduction`, `_pronation`, `_ulnarDeviation`, â€¦) |
| Total columns | 281 | 241 |
| Coordinates | world frame, metres, hand at 0 | different axes, scale and placement, and **mirrored** (opposite handedness) |

Because of these differences the model uses neither raw positions nor rotations. `data-capture/features.py` turns the positions into 114 hand-local, size-normalised bone positions (Studio z flipped to undo the mirroring). Those are the same for both sources.
