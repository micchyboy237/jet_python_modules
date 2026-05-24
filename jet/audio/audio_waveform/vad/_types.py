from pathlib import Path
from typing import List, Literal, Optional, TypedDict, Union

import numpy as np
from jet.audio.speech.vad_types import ValleyTrough

AudioInput = Union[np.ndarray, bytes, bytearray, str, Path]


class SpeechWaveMeta(TypedDict):
    has_risen: bool
    has_multi_passed: bool
    has_fallen: bool
    is_valid: bool


class SpeechWaveDetails(TypedDict):
    """Detailed insights including frame boundaries and probability statistics for a speech wave."""

    frame_start: int
    frame_end: int
    frame_len: int
    duration_sec: float
    min_prob: float
    max_prob: float
    avg_prob: float
    std_prob: float


class SpeechWave(SpeechWaveMeta):
    start_sec: float
    end_sec: float
    details: SpeechWaveDetails


SpeechEndReason = Literal["silence", "valley", "hard_limit"]


class _SpeechSegmentRequired(TypedDict):
    num: int
    start: float | int
    end: float | int
    prob: float
    duration: float
    frames_length: int
    frame_start: int
    frame_end: int
    type: Literal["speech", "non-speech"]
    is_ongoing: bool  # true for final open/ongoing segment in streaming/full audio
    segment_probs: List[float]


class SpeechSegment(_SpeechSegmentRequired):
    last_non_speech_sec: Optional[float]  # duration of trailing silence with energy
    end_reason: Optional[SpeechEndReason]  # only this key is optional
    best_valley_trough: Optional[
        ValleyTrough
    ]  # trough that caused this segment's split, if any
    # Absolute timestamps as ISO 8601 strings (UTC)
    # Example: "2026-05-25T14:32:17.123456+00:00"
    start_time_utc: Optional[str]
    end_time_utc: Optional[str]


class WordSegment(TypedDict):
    index: int
    start_ms: Optional[int]
    end_ms: Optional[int]
    word: Optional[str]
