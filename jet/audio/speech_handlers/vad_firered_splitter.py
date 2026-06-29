"""
vad_firered_splitter.py — Utility to split speech segments using FireRedVAD.

Provides ``split_segment_with_vad()`` which runs a secondary VAD pass on
a speech segment's audio and returns independent sub-segments with properly
adjusted timestamps. Each sub-segment is a standalone ``SpeechSegment``
ready to be saved and dispatched through the normal pipeline.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.audio_waveform.vad.vad_firered import (
    DEFAULT_MAX_BUFFER_SEC,
    DEFAULT_MAX_SPEECH_SEC,
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_MIN_SPEECH_SEC,
    DEFAULT_PAD_START_FRAME,
    DEFAULT_SMOOTH_WINDOW_SIZE,
    DEFAULT_THRESHOLD,
    extract_speech_timestamps,
)
from jet.audio.helpers.base import get_audio_duration
from jet.logger import logger


def split_segment_with_vad(
    segment: SpeechSegment,
    audio_np: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_speech_duration_sec: Optional[float] = DEFAULT_MAX_SPEECH_SEC,
    smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
    pad_start_frame: int = DEFAULT_PAD_START_FRAME,
    max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
    min_sub_segment_duration_sec: float = 0.1,
    verbose: bool = False,
) -> List[SpeechSegment]:
    """
    Run FireRedVAD on a speech segment's audio and split it into independent
    sub-segments. Each sub-segment has properly adjusted global timestamps
    and its own per-frame ``segment_probs`` sliced from the VAD output.

    Parameters
    ----------
    segment:
        The original SpeechSegment (with global start/end timestamps).
    audio_np:
        The audio data for this segment (numpy array).
    sample_rate:
        Sample rate of the audio.
    threshold, min_silence_duration_sec, min_speech_duration_sec, etc.:
        VAD parameters (same defaults as vad_firered.extract_speech_timestamps).
    min_sub_segment_duration_sec:
        Minimum duration for a sub-segment to be kept.

    Returns
    -------
    List[SpeechSegment]
        List of independent sub-segments with:
        - adjusted global timestamps
        - per-frame ``segment_probs`` (sliced from the VAD output)
        Returns a single-element list with the original segment if:
        - Audio is empty
        - Segment is too short to split
        - VAD fails
        - VAD finds no sub-segments
        - All sub-segments are filtered out
    """
    if audio_np is None or audio_np.size == 0:
        logger.warning(
            "split_segment_with_vad: empty audio for segment %s — returning as-is",
            segment.get("num"),
        )
        return [segment]

    duration_sec = get_audio_duration(audio_np, sample_rate)

    # Skip segments that are too short to meaningfully split
    if duration_sec < min_speech_duration_sec * 2:
        logger.debug(
            "split_segment_with_vad: segment %s too short (%.3fs) for splitting — "
            "returning as-is",
            segment.get("num"),
            duration_sec,
        )
        return [segment]

    logger.info(
        "split_segment_with_vad: running secondary VAD on segment %s (%.3fs, %d samples)",
        segment.get("num"),
        duration_sec,
        len(audio_np),
    )

    # -------------------------------------------------------------------
    # Run FireRedVAD with with_scores=True to get per-frame probabilities.
    # These are sliced per sub-segment so save_segment gets real data.
    # -------------------------------------------------------------------
    try:
        sub_segments, probs = extract_speech_timestamps(
            audio=audio_np,
            threshold=threshold,
            min_silence_duration_sec=min_silence_duration_sec,
            min_speech_duration_sec=min_speech_duration_sec,
            max_speech_duration_sec=max_speech_duration_sec,
            return_seconds=True,
            with_scores=True,  # ← CHANGED: get per-frame probs
            include_non_speech=False,
            smooth_window_size=smooth_window_size,
            pad_start_frame=pad_start_frame,
            max_buffer_sec=max_buffer_sec,
        )
    except Exception as exc:
        logger.error(
            "split_segment_with_vad: VAD failed for segment %s: %s — returning as-is",
            segment.get("num"),
            exc,
        )
        return [segment]

    if not sub_segments:
        logger.info(
            "split_segment_with_vad: no sub-segments found in segment %s — "
            "returning as-is",
            segment.get("num"),
        )
        return [segment]

    # Filter out sub-segments that are too short
    valid_sub_segments: List[SpeechSegment] = [
        seg for seg in sub_segments if seg["duration"] >= min_sub_segment_duration_sec
    ]

    if not valid_sub_segments:
        logger.info(
            "split_segment_with_vad: all sub-segments filtered out "
            "(min_duration=%.3fs) in segment %s — returning as-is",
            min_sub_segment_duration_sec,
            segment.get("num"),
        )
        return [segment]

    # If only one sub-segment and it covers the whole audio, no point splitting
    if len(valid_sub_segments) == 1:
        only_seg = valid_sub_segments[0]
        if (
            float(only_seg["start"]) < 0.05
            and float(only_seg["end"]) > duration_sec - 0.05
        ):
            logger.info(
                "split_segment_with_vad: single sub-segment covers entire audio "
                "in segment %s — returning as-is",
                segment.get("num"),
            )
            return [segment]

    logger.info(
        "split_segment_with_vad: segment %s split into %d sub-segments "
        "(filtered from %d total)",
        segment.get("num"),
        len(valid_sub_segments),
        len(sub_segments),
    )

    # -------------------------------------------------------------------
    # Build independent SpeechSegments with adjusted timestamps and
    # sliced per-frame probabilities.
    # -------------------------------------------------------------------
    original_start_sec = float(segment["start"])
    original_start_utc = segment.get("start_time_utc")
    had_overflow = segment.get("had_overflow", False)

    # Parse original UTC start time once
    orig_start_dt: Optional[datetime] = None
    if original_start_utc is not None:
        try:
            orig_start_dt = datetime.fromisoformat(str(original_start_utc))
            if orig_start_dt.tzinfo is None:
                orig_start_dt = orig_start_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError) as exc:
            logger.debug(
                "split_segment_with_vad: could not parse start_time_utc=%s: %s",
                original_start_utc,
                exc,
            )

    # Pre-compute hop_sec for slicing probs (matches vad_firered: 10ms frames)
    hop_sec = 0.010

    result_segments: List[SpeechSegment] = []
    for sub_seg in valid_sub_segments:
        sub_start_sec = float(sub_seg["start"])
        sub_end_sec = float(sub_seg["end"])
        sub_duration = sub_end_sec - sub_start_sec

        # Extract audio for this sub-segment
        start_sample = int(round(sub_start_sec * sample_rate))
        end_sample = int(round(sub_end_sec * sample_rate))
        sub_audio = audio_np[start_sample:end_sample].copy()

        if sub_audio.size == 0:
            logger.warning(
                "split_segment_with_vad: sub-segment [%.3fs, %.3fs] of segment %s "
                "has empty audio — skipping",
                sub_start_sec,
                sub_end_sec,
                segment.get("num"),
            )
            continue

        # Slice per-frame probabilities for this sub-segment
        frame_start_idx = int(sub_start_sec / hop_sec)
        frame_end_idx = int(sub_end_sec / hop_sec)
        sub_probs = probs[frame_start_idx : frame_end_idx + 1] if probs else []

        # Create new SpeechSegment with global timestamps and probs
        new_seg = copy.deepcopy(segment)
        new_seg["start"] = original_start_sec + sub_start_sec
        new_seg["end"] = original_start_sec + sub_end_sec
        new_seg["duration"] = sub_duration
        new_seg.pop("num", None)  # assigned by SegmentStore
        new_seg["segment_probs"] = sub_probs  # ← sliced per-frame probs
        new_seg["had_overflow"] = had_overflow

        # Also update prob (average) and frames_length for consistency
        if sub_probs:
            new_seg["prob"] = float(np.mean(sub_probs))
            new_seg["frames_length"] = len(sub_probs)
            new_seg["frame_start"] = frame_start_idx
            new_seg["frame_end"] = frame_end_idx

        # Adjust UTC timestamps
        if orig_start_dt is not None:
            new_start_ts = orig_start_dt.timestamp() + sub_start_sec
            new_start_dt = datetime.fromtimestamp(new_start_ts, tz=timezone.utc)
            new_seg["start_time_utc"] = new_start_dt.isoformat()

            new_end_ts = new_start_dt.timestamp() + sub_duration
            new_end_dt = datetime.fromtimestamp(new_end_ts, tz=timezone.utc)
            new_seg["end_time_utc"] = new_end_dt.isoformat()
        else:
            new_seg.pop("start_time_utc", None)
            new_seg.pop("end_time_utc", None)

        if verbose:
            logger.debug(
                "split_segment_with_vad: created sub-segment [%.3fs → %.3fs global] "
                "(dur=%.3fs, %d samples, %d probs)",
                float(new_seg["start"]),
                float(new_seg["end"]),
                sub_duration,
                len(sub_audio),
                len(sub_probs),
            )

        result_segments.append(new_seg)

    if not result_segments:
        logger.warning(
            "split_segment_with_vad: all sub-segments had empty audio — "
            "returning original segment %s",
            segment.get("num"),
        )
        return [segment]

    return result_segments
