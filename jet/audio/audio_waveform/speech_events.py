# jet_python_modules/jet/audio/audio_waveform/speech_events.py
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SpeechSegmentStartEvent:
    segment_id: int
    start_frame: int
    start_time_sec: float
    datetime_started: str
    segment_dir: Path | None = None  # to be set by a handler if desired


@dataclass
class SpeechSegmentEndEvent:
    segment_id: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    audio: np.ndarray
    probs: list[dict]
    forced_split: bool
    trigger_reason: str
    summary: dict
    segment_dir: Path | None = None
