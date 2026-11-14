"""Thread-safe ring buffer of stream frames (one timestamp + one value row per frame)."""
from __future__ import annotations

import threading
from typing import Sequence

import numpy as np


class FrameBuffer:
    def __init__(self, columns: Sequence[str], maxlen: int):
        self.columns = list(columns)
        self.maxlen = maxlen
        self._data = np.full((maxlen, len(self.columns)), np.nan, dtype=np.float32)
        self._ts = np.zeros(maxlen, dtype=np.float64)   # seconds
        self._head = 0          # next write index
        self._size = 0          # valid frames
        self._lock = threading.Lock()

    def push(self, timestamp: float, values: Sequence[float] | np.ndarray) -> None:
        with self._lock:
            self._data[self._head] = values
            self._ts[self._head] = timestamp
            self._head = (self._head + 1) % self.maxlen
            self._size = min(self._size + 1, self.maxlen)

    def clear(self) -> None:
        with self._lock:
            self._head = self._size = 0

    def __len__(self) -> int:
        return self._size

    def latest(self, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(timestamps[n], values[n, columns]) oldest -> newest, as copies."""
        with self._lock:
            n = self._size if n is None else min(n, self._size)
            idx = np.arange(self._head - n, self._head) % self.maxlen
            return self._ts[idx], self._data[idx]

    def last_seconds(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        """Frames within `seconds` of the newest frame."""
        with self._lock:
            idx = np.arange(self._head - self._size, self._head) % self.maxlen
            ts = self._ts[idx]
            if len(ts) == 0:
                return ts, self._data[idx]
            idx = idx[np.searchsorted(ts, ts[-1] - seconds):]
            return self._ts[idx], self._data[idx]

    def to_dataframe(self):
        import pandas as pd

        ts, data = self.latest()
        df = pd.DataFrame(data, columns=self.columns)
        df.insert(0, "Timestamp", ts)
        return df
