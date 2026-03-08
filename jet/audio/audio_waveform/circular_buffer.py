from collections import deque
from typing import Deque

import numpy as np


class CircularBuffer:
    """Fixed-length circular buffer for scalar values."""

    def __init__(self, max_len: int) -> None:
        if max_len <= 0:
            raise ValueError("max_len must be > 0")

        self.max_len = max_len
        self._buffer: Deque[float] = deque(maxlen=max_len)

    def append(self, values: np.ndarray | float) -> None:
        if isinstance(values, np.ndarray):
            for v in values:
                self._buffer.append(float(v))
        else:
            self._buffer.append(float(values))

    def to_array(self) -> np.ndarray:
        return np.array(self._buffer, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._buffer)
