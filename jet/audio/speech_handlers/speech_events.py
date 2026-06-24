"""Speech segment event types for the handler pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.helpers.config import SAMPLE_RATE


@dataclass
class SpeechSegmentStartEvent:
    """Fired when a speech segment begins (VAD detects speech start)."""

    segment: SpeechSegment
    segment_number: int
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sample_rate: int = SAMPLE_RATE


@dataclass
class SpeechSegmentEndEvent:
    """Fired when a speech segment ends (VAD detects silence/timeout).

    NEW: enqueued_at records when the event was created, used by
    WebsocketSubtitleSender to detect stale segments.
    """

    segment: SpeechSegment
    segment_number: int
    audio_np: np.ndarray
    segment_dir: Path
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sample_rate: int = SAMPLE_RATE
    enqueued_at: float = field(default_factory=time.monotonic)
