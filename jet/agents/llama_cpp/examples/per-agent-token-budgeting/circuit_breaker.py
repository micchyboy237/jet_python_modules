"""
Per-agent circuit breaker.

If a single agent fails or gets rejected (budget/rate-limit errors count as
failures here) repeatedly, trip its breaker: stop letting it call the model
for a cooldown period. This protects the shared llama.cpp server from a
misbehaving agent hammering it with retries, and protects the rest of the
fleet's latency/throughput from one agent's problem.

States: closed (normal) -> open (blocking calls) -> half_open (one probe
call allowed) -> closed (if probe succeeds) or open again (if it fails).
"""

from __future__ import annotations

import threading
import time
from enum import Enum


class CircuitOpenError(Exception):
    pass


class State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown_s: float = 20.0):
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._state = State.CLOSED
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def before_call(self) -> None:
        with self._lock:
            if self._state == State.OPEN:
                if time.monotonic() - self._opened_at >= self.cooldown_s:
                    self._state = State.HALF_OPEN
                else:
                    raise CircuitOpenError(
                        f"circuit open, retry after "
                        f"{self.cooldown_s - (time.monotonic() - self._opened_at):.1f}s"
                    )
            # CLOSED or HALF_OPEN: allow the call through

    def on_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = State.CLOSED

    def on_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == State.HALF_OPEN or self._failures >= self.failure_threshold:
                self._state = State.OPEN
                self._opened_at = time.monotonic()

    @property
    def state(self) -> State:
        with self._lock:
            return self._state
