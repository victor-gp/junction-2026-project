"""Read solved Smartgloves hand poses from the Rokoko Device SDK into a FrameBuffer.

    gloves -> rokoko-sdk (driver, raw sensors)  -> rkk-hand-solver --emit both (solved hands)
           -> GloveStreamReader (RGMP v2 over TCP) -> FrameBuffer

The reader starts rkk-hand-solver itself (config.START_SOLVER); the solver starts
the driver unless one is already running.

One buffer row per solved frame, both hands, with the bone/joint names and column
order of the Rokoko Studio CSV export:
    <Bone>_position_{x,y,z}     world position of the bone (metres; hand itself at 0)
    <Joint>_rotation_{x,y,z,w}  rotation relative to the parent bone (quaternion)
A hand that is not connected is NaN.
"""
from __future__ import annotations

import json
import re
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field

import numpy as np

import config
from buffer import FrameBuffer

# ── RGMP v2 protocol (as in the SDK's examples/python/rgmp_stream.py) ────────
MSG_DEFINITION, MSG_DATA, MSG_DISCONNECT = 1, 2, 3
OUTER_HEADER = struct.Struct("<II")       # msg_prefix, msg_len
DATA_HEADER = struct.Struct("<IIQ")       # device_id, group_id, timestamp_us
_TYPE_SIZES = {"INT32": 4, "UINT32": 4, "INT64": 8, "UINT64": 8, "FLOAT": 4, "DOUBLE": 8}
_DATA_TYPE_RE = re.compile(r"^(INT32|UINT32|INT64|UINT64|FLOAT|DOUBLE)(?:\[(\d+)(?:,\s*(\d+))?\])?$")


def _data_type(data_type: str) -> tuple[str, int]:
    """('FLOAT', byte_size) for an RGMP data_type like 'FLOAT[7]'."""
    m = _DATA_TYPE_RE.match(data_type.strip())
    if not m:
        raise ValueError(f"Unrecognized data_type: {data_type!r}")
    return m.group(1), _TYPE_SIZES[m.group(1)] * int(m.group(2) or 1) * int(m.group(3) or 1)


# ── Solver joints -> Studio names and columns ────────────────────────────────
# solver frame (without side prefix) -> (Studio bone for position, Studio joint for rotation)
JOINT_MAP: dict[str, tuple[str, str]] = {
    "hand": ("Hand", "Wrist"),
    "finger1_metacarpal": ("Digit1MetaCarpal", "Digit1Carpometacarpal"),
    "finger1_proximal": ("Digit1ProximalPhalangx", "Digit1Metacarpophalangeal"),
    "finger1_distal": ("Digit1DistalPhalanx", "Digit1Interphalangeal"),
    **{k: v for d in range(2, 6) for k, v in {
        f"finger{d}_metacarpal": (f"Digit{d}MetaCarpal", f"Digit{d}Carpometacarpal"),
        f"finger{d}_proximal": (f"Digit{d}ProximalPhalangx", f"Digit{d}Metacarpophalangeal"),
        f"finger{d}_medial": (f"Digit{d}IntermediatePhalanx", f"Digit{d}ProximalInterphalangeal"),
        f"finger{d}_distal": (f"Digit{d}DistalPhalanx", f"Digit{d}DistalInterphalangeal"),
    }.items()},
}
# Studio order: both hands first, then left digits, then right digits. 7 columns per joint.
_ORDER = [(s, "hand") for s in ("left", "right")] + \
         [(s, k) for s in ("left", "right") for k in JOINT_MAP if k != "hand"]
COLUMN_INDEX = {key: 7 * i for i, key in enumerate(_ORDER)}
COLUMNS = [
    name
    for side, key in _ORDER
    for name in (
        [f"{side.capitalize()}{JOINT_MAP[key][0]}_position_{a}" for a in "xyz"]
        + [f"{side.capitalize()}{JOINT_MAP[key][1]}_rotation_{a}" for a in "xyzw"]
    )
]


@dataclass(slots=True)
class _Slot:
    offset: int      # byte offset inside the group payload
    count: int       # 3 = position, 4 = rotation
    column: int      # first target column in the row


@dataclass
class _Hand:
    side: str
    groups: dict[int, list[_Slot]]
    ts: int = -1
    got: set[int] = field(default_factory=set[int])


def _parse_definition(definition: dict) -> _Hand | None:
    """Decoding plan for one solved hand; None for anything else."""
    side = str((definition.get("device_info") or {}).get("hand", "")).lower()
    if definition.get("device_type") != "solved_hand" or side not in ("left", "right"):
        return None
    groups: dict[int, list[_Slot]] = {}
    for gid, group in enumerate(definition.get("groups", [])):
        offset, slots = 0, []
        for stream in group.get("streams", []):
            base, size = _data_type(stream["data_type"])
            col = COLUMN_INDEX.get((side, str(stream.get("target_frame", "")).removeprefix(f"{side}_")))
            measure = stream.get("measure_type")
            if col is not None and base == "FLOAT":
                if measure == "TRANSFORM" and stream.get("reference_frame") == "LTP_NED":
                    slots.append(_Slot(offset, 3, col))        # world position
                elif measure == "ORIENTATION":
                    slots.append(_Slot(offset, 4, col + 3))    # rotation relative to parent
            offset += size
        if slots:
            groups[gid] = slots
    return _Hand(side, groups) if groups else None


def _port_open(addr: tuple[str, int]) -> bool:
    try:
        socket.create_connection(addr, timeout=0.5).close()
        return True
    except OSError:
        return False


class GloveStreamReader:
    """reader = GloveStreamReader().start(); reader.buffer ...; reader.stop()  (or use `with`)."""

    def __init__(self, start_solver: bool | None = None):
        self.start_solver = config.START_SOLVER if start_solver is None else start_solver
        self.addr = config.SOLVER_ADDR
        self.buffer = FrameBuffer(COLUMNS, int(config.STREAM_RATE_HZ * config.BUFFER_SECONDS))
        self.last_error: str | None = None
        self._hands: dict[int, _Hand] = {}          # device_id -> hand
        self._ready: set[int] = set()               # hands with a complete sample since last push
        self._row = np.full(len(COLUMNS), np.nan, dtype=np.float32)
        self._sock: socket.socket | None = None
        self._solver: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None

    @property
    def connected_hands(self) -> list[str]:
        return sorted(h.side for h in self._hands.values())

    # ── control ──────────────────────────────────────────────────────────────
    def start(self) -> "GloveStreamReader":
        if self.start_solver:
            args = [config.SOLVER_EXE, "--emit", "both", "--listen", "%s:%d" % self.addr]
            args += ["--no-driver"] if _port_open(config.DRIVER_ADDR) else ["--driver-path", config.DRIVER_EXE]
            self._solver = subprocess.Popen(args)
        deadline = time.monotonic() + config.CONNECT_TIMEOUT_S
        while self._sock is None:
            try:
                self._sock = socket.create_connection(self.addr, timeout=1.0)
            except OSError:
                if time.monotonic() > deadline or (self._solver and self._solver.poll() is not None):
                    self.stop()
                    raise ConnectionError("hand solver not reachable on %s:%d" % self.addr)
                time.sleep(0.5)
        self._sock.settimeout(None)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._thread = threading.Thread(target=self._run, name="rgmp-reader", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if self._thread:
            self._thread.join(timeout=2)
        if self._solver is not None:
            self._solver.terminate()
            try:
                self._solver.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._solver.kill()
            self._solver = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ── reading ──────────────────────────────────────────────────────────────
    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray(n)
        view, got = memoryview(buf), 0
        while got < n:
            sock = self._sock
            if sock is None:
                raise ConnectionError("reader stopped")
            r = sock.recv_into(view[got:], n - got)
            if r == 0:
                raise ConnectionError("server closed the connection")
            got += r
        return bytes(buf)

    def _run(self) -> None:
        try:
            while True:
                prefix, length = OUTER_HEADER.unpack(self._recv_exact(OUTER_HEADER.size))
                payload = self._recv_exact(length)
                if prefix == MSG_DEFINITION:
                    self._on_definition(json.loads(payload.decode("utf-8")))
                elif prefix == MSG_DATA:
                    self._on_data(payload)
                elif prefix == MSG_DISCONNECT and len(payload) >= 4:
                    self._on_disconnect(struct.unpack_from("<I", payload)[0])
                # unknown frame types are skipped, per the protocol
        except ConnectionError as e:
            if self._sock is not None:
                self.last_error = str(e)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"

    def _on_definition(self, definition: dict) -> None:
        hand = _parse_definition(definition)
        if hand is not None:
            self._on_disconnect(definition["device_id"])
            self._hands[definition["device_id"]] = hand

    def _on_disconnect(self, dev: int) -> None:
        hand = self._hands.pop(dev, None)
        self._ready.discard(dev)
        if hand is not None:
            for key in JOINT_MAP:
                c = COLUMN_INDEX[(hand.side, key)]
                self._row[c:c + 7] = np.nan
            if self._ready and self._ready >= self._hands.keys():
                self._push()   # remaining hands were only waiting for this one

    def _on_data(self, payload: bytes) -> None:
        if len(payload) < DATA_HEADER.size:
            return
        dev, gid, ts_us = DATA_HEADER.unpack_from(payload)
        hand = self._hands.get(dev)
        if hand is None or gid not in hand.groups:
            return
        if ts_us != hand.ts:
            # New sample for this hand; if its previous one is still waiting
            # for the other hand, emit that row first instead of overwriting it.
            if dev in self._ready:
                self._push()
            hand.ts, hand.got = ts_us, set()
        for s in hand.groups[gid]:
            self._row[s.column:s.column + s.count] = struct.unpack_from(
                f"<{s.count}f", payload, DATA_HEADER.size + s.offset)
        hand.got.add(gid)
        if hand.got == hand.groups.keys():
            self._ready.add(dev)
            if self._ready >= self._hands.keys():
                self._push()

    def _push(self) -> None:
        ts = max(self._hands[d].ts for d in self._ready)
        self.buffer.push(ts / 1e6, self._row)
        self._ready.clear()
