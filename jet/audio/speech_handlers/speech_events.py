"""
Typed event objects passed between the recorder and speech segment handlers.
Kept deliberately thin — handlers pull what they need, nothing is computed here.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechSegment


@dataclass
class SpeechSegmentStartEvent:
    """Fired when a new speech segment has been detected but not yet completed."""

    segment: SpeechSegment
    segment_number: int
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SpeechSegmentEndEvent:
    """Fired when a speech segment is fully captured and ready for processing."""

    segment: SpeechSegment
    segment_number: int
    audio_np: np.ndarray  # raw segment audio (int16 or float32)
    segment_dir: Path  # path to segment_NNN/ directory
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sample_rate: int = 16000
