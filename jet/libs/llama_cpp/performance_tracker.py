from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class PerformanceMetrics:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft: Optional[float]
    decode_speed: Optional[float]
    overall_speed: Optional[float]
    total_latency: float


class PerformanceTracker:
    """
    Tracks streaming performance metrics:
    - Time To First Token (TTFT)
    - Decode speed (tokens/sec, generation only)
    - Overall speed (tokens/sec, wall-clock)
    - Total latency
    """

    def __init__(self) -> None:
        self.start_time = time.perf_counter()
        self.first_token_time: Optional[float] = None
        self.last_token_time: Optional[float] = None
        self.token_count = 0

    def mark_token(self) -> None:
        """
        Call this every time a token (or chunk with content) is received.
        """
        now = time.perf_counter()

        if self.first_token_time is None:
            self.first_token_time = now

        self.last_token_time = now
        self.token_count += 1

    def finalize(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> PerformanceMetrics:
        """
        Finalize metrics using OpenAI usage data.
        """
        end_time = time.perf_counter()
        total_latency = end_time - self.start_time

        # TTFT
        ttft: Optional[float] = None
        if self.first_token_time is not None:
            ttft = self.first_token_time - self.start_time

        # Decode speed (generation only)
        decode_speed: Optional[float] = None
        if (
            self.first_token_time is not None
            and self.last_token_time is not None
            and completion_tokens > 0
        ):
            generation_duration = self.last_token_time - self.first_token_time
            if generation_duration > 0:
                decode_speed = completion_tokens / generation_duration

        # Overall speed (wall-clock)
        overall_speed: Optional[float] = None
        if completion_tokens > 0 and total_latency > 0:
            overall_speed = completion_tokens / total_latency

        return PerformanceMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            ttft=ttft,
            decode_speed=decode_speed,
            overall_speed=overall_speed,
            total_latency=total_latency,
        )
