"""
Token bucket for rate-limiting token throughput per agent (tokens-per-minute
style), independent from the *cumulative spend* budget in token_budget.py.

Why both exist:
- token_budget.py answers "has this agent used more tokens, total, than
  it's allowed to ever use in this run?"
- TokenBucket here answers "is this agent trying to burn tokens faster than
  its allowed rate right now?" -- it catches a tight retry loop or a burst
  of parallel calls from one agent even if that agent is nowhere near its
  total budget yet.

Refill is continuous (not fixed-window), so an agent can't dump its entire
allowance right at a window boundary and effectively double its rate.
"""

from __future__ import annotations

import threading
import time


class RateLimitedError(Exception):
    pass


class TokenBucket:
    def __init__(self, capacity: int, refill_per_second: float):
        """
        capacity:          max tokens the bucket can hold (i.e. max burst size)
        refill_per_second: sustained tokens-per-minute / 60
        """
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_second)
        self._last_refill = now

    def try_consume(self, amount: int) -> bool:
        """Non-blocking: returns False immediately if not enough tokens."""
        with self._lock:
            self._refill()
            if self._tokens >= amount:
                self._tokens -= amount
                return True
            return False

    def consume_or_wait(self, amount: int, max_wait_s: float = 30.0) -> None:
        """Blocking: waits for capacity to free up, or raises if it would
        take longer than max_wait_s."""
        deadline = time.monotonic() + max_wait_s
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                deficit = amount - self._tokens
                wait_needed = deficit / self.refill_per_second
            if time.monotonic() + wait_needed > deadline:
                raise RateLimitedError(
                    f"Would need to wait {wait_needed:.1f}s for {amount} tokens, "
                    f"exceeding max_wait_s={max_wait_s}"
                )
            time.sleep(min(wait_needed, 0.5))

    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens
