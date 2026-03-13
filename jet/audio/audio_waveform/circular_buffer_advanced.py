import numpy as np


class CircularBuffer:
    """Fixed-size circular buffer optimized for streaming audio samples."""

    def __init__(self, capacity: int, dtype=np.float32):
        self.capacity = int(capacity)
        self.buffer = np.zeros(self.capacity, dtype=dtype)
        self.write_pos = 0
        self.read_pos = 0
        self.count = 0

    def extend(self, data: np.ndarray) -> None:
        if len(data) == 0:
            return
        n = len(data)

        if self.write_pos + n <= self.capacity:
            self.buffer[self.write_pos : self.write_pos + n] = data
            self.write_pos += n
        else:
            first = self.capacity - self.write_pos
            self.buffer[self.write_pos :] = data[:first]
            remaining = n - first
            self.buffer[:remaining] = data[first : first + remaining]
            self.write_pos = remaining

        self.count = min(self.count + n, self.capacity)

    def get_frame(self, length: int) -> np.ndarray | None:
        if self.available() < length:
            return None

        if self.read_pos + length <= self.capacity:
            return self.buffer[self.read_pos : self.read_pos + length].copy()
        else:
            part1 = self.capacity - self.read_pos
            frame = np.empty(length, dtype=self.buffer.dtype)
            frame[:part1] = self.buffer[self.read_pos :]
            frame[part1:] = self.buffer[: length - part1]
            return frame

    def advance(self, n: int) -> None:
        if n <= 0 or n > self.count:
            n = min(n, self.count)
        self.read_pos = (self.read_pos + n) % self.capacity
        self.count -= n

    def available(self) -> int:
        return self.count

    def clear(self) -> None:
        self.write_pos = 0
        self.read_pos = 0
        self.count = 0
        self.buffer.fill(0)
