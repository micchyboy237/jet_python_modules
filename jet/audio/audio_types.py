from __future__ import annotations

import os
from typing import List, Literal, Optional, TypedDict, Union

import numpy as np
import numpy.typing as npt
import torch

AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    torch.Tensor,
]


class SpeechWaveMeta(TypedDict):
    has_risen: bool
    has_multi_passed: bool
    has_fallen: bool
    is_valid: bool


class MergedWaveInfo(TypedDict):
    """Snapshot of one constituent wave before it was absorbed into a merge."""

    frame_start: int
    frame_end: int
    start_sec: float
    end_sec: float
    duration_sec: float
    max_prob: float
    prominence: float


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
    avg_hybrid: float
    # Hybrid gate bookkeeping
    rms_hold_frames: int
    merge_count: int  # how many raw waves were fused to make this one (0 = no merge)
    merged: bool
    merged_waves: List[MergedWaveInfo]


class SpeechWave(SpeechWaveMeta):
    start_sec: float
    end_sec: float
    details: SpeechWaveDetails


class SpeechSegment(TypedDict):
    num: int
    start: float | int
    end: float | int
    prob: float
    duration: float
    frames_length: int
    frame_start: int
    frame_end: int
    type: Literal["speech", "non-speech"]
    segment_probs: List[float]


class WordSegment(TypedDict):
    index: int
    start_ms: Optional[int]
    end_ms: Optional[int]
    word: Optional[str]
