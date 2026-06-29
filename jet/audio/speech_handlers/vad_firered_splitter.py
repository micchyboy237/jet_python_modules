"""
vad_firered_splitter.py — Utility to split speech segments using FireRedVAD.
Provides ``split_segment_with_vad()`` which runs a secondary VAD pass on
a speech segment's audio and returns independent sub-segments with properly
adjusted timestamps. Each sub-segment is a standalone ``SpeechSegment``
ready to be saved and dispatched through the normal pipeline.

Enhanced with minimum duration enforcement: sub-segments shorter than
``min_sub_segment_duration_sec`` are intelligently merged back into
adjacent segments, **but only when the sub-segment's ``end_reason`` is
``"valley"``** (i.e., the split occurred at a natural speech pause).
This prevents conflicts with the short-segment accumulation logic in
``ShortSegmentAccumulator``, which handles segments ending due to
``hard_limit``, ``silence``, or other reasons.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_MAX_SPEECH_SEC,
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_MIN_SPEECH_SEC,
    DEFAULT_MIN_SUB_SEG_DURATION_SEC,
    DEFAULT_SOFT_LIMIT_SEC,
    DEFAULT_THRESHOLD,
)
from jet.audio.audio_waveform.vad.vad_firered import (
    DEFAULT_MAX_BUFFER_SEC,
    DEFAULT_PAD_START_FRAME,
    DEFAULT_SMOOTH_WINDOW_SIZE,
    extract_speech_timestamps,
)
from jet.audio.helpers.base import get_audio_duration
from jet.audio.normalization.norm_speech_loudness import normalize_audio_for_vad
from jet.audio.normalization.quant import quantize_audio
from jet.logger import logger


def _should_consider_for_merging(
    segment: SpeechSegment, min_duration_sec: float
) -> bool:
    """
    Determine if a short segment should be considered for merging.

    Only segments with ``end_reason == "valley"`` are eligible for merging.
    Segments ending due to ``hard_limit``, ``silence``, or ``None`` are
    left for the downstream ``ShortSegmentAccumulator`` to handle.

    Args:
        segment: The sub-segment to check
        min_duration_sec: Minimum duration threshold

    Returns:
        True if the segment is short AND has valley end_reason
    """
    duration = float(segment.get("duration", 0.0))
    if duration >= min_duration_sec:
        return False

    end_reason = segment.get("end_reason")
    is_valley = end_reason == "valley"

    if not is_valley:
        logger.debug(
            "_should_consider_for_merging: segment [%.3fs → %.3fs] dur=%.3fs "
            "is short but end_reason=%s (not 'valley') — skipping merge, "
            "delegating to ShortSegmentAccumulator",
            float(segment["start"]),
            float(segment["end"]),
            duration,
            end_reason,
        )

    return is_valley


def _merge_short_valley_segments(
    segments: List[SpeechSegment],
    min_duration_sec: float,
    audio_np: np.ndarray,
    sample_rate: int,
    original_start_sec: float,
    had_overflow: bool,
    orig_start_dt: Optional[datetime],
    probs: List[float],
    hop_sec: float,
    verbose: bool = False,
) -> List[SpeechSegment]:
    """
    Merge sub-segments that are shorter than min_duration_sec **AND**
    have ``end_reason == "valley"`` into adjacent segments.

    Strategy:
    1. Identify short valley segments (< min_duration_sec AND end_reason="valley")
    2. For each short valley segment, decide whether to merge with left, right, or both neighbors
    3. Merge criteria:
       - If no left neighbor, merge with right
       - If no right neighbor, merge with left
       - If both exist, merge with the one that has the smaller gap OR is shorter
    4. Build complete merge groups BEFORE emitting any segments (prevents double emission)
    5. After merging, re-check for any remaining short valley segments recursively

    Non-valley short segments (end_reason = hard_limit, silence, None)
    are left untouched and will be handled by ShortSegmentAccumulator downstream.

    Args:
        segments: List of sub-segments (with absolute timestamps already adjusted)
        min_duration_sec: Minimum allowed duration for a segment
        audio_np: Original audio array for the parent segment
        sample_rate: Audio sample rate
        original_start_sec: Absolute start time of the parent segment
        had_overflow: Whether the parent segment had overflow
        orig_start_dt: Parsed start datetime of the parent segment
        probs: Full probability array from VAD
        hop_sec: Hop size in seconds (0.010)
        verbose: Enable debug logging

    Returns:
        List of merged sub-segments
    """
    if len(segments) <= 1:
        return segments

    # ── Step 1: Identify short valley segments eligible for merging ────────
    short_valley_indices = [
        i
        for i, seg in enumerate(segments)
        if _should_consider_for_merging(seg, min_duration_sec)
    ]

    if not short_valley_indices:
        if verbose:
            total_short = sum(
                1
                for seg in segments
                if float(seg.get("duration", 0.0)) < min_duration_sec
            )
            logger.debug(
                "_merge_short_valley_segments: %d segments, %d short (< %.3fs), "
                "but none are valley-split — nothing to merge",
                len(segments),
                total_short,
                min_duration_sec,
            )
        return segments

    if verbose:
        total_short = sum(
            1 for seg in segments if float(seg.get("duration", 0.0)) < min_duration_sec
        )
        logger.debug(
            "_merge_short_valley_segments: %d segments total, %d short (< %.3fs), "
            "%d short-valley (eligible for merge)",
            len(segments),
            total_short,
            min_duration_sec,
            len(short_valley_indices),
        )
        for idx in short_valley_indices:
            seg = segments[idx]
            logger.debug(
                "  short-valley seg[%d]: [%.3fs → %.3fs] dur=%.3fs end_reason=%s",
                idx,
                float(seg["start"]),
                float(seg["end"]),
                float(seg.get("duration", 0.0)),
                seg.get("end_reason"),
            )

    # ── Step 2: Decide merge direction for each short valley segment ──────
    merge_decisions = {}  # {index: "left" | "right"}

    for idx in short_valley_indices:
        # Find valid left neighbor (not itself a short-valley being merged away)
        has_left = False
        left_idx = idx - 1
        while left_idx >= 0:
            if left_idx not in merge_decisions:
                has_left = True
                break
            left_idx -= 1

        # Find valid right neighbor
        has_right = False
        right_idx = idx + 1
        while right_idx < len(segments):
            if right_idx not in merge_decisions:
                has_right = True
                break
            right_idx += 1

        if not has_left and not has_right:
            if verbose:
                logger.debug(
                    "  seg[%d]: isolated short-valley segment - keeping as-is",
                    idx,
                )
            continue

        if has_left and not has_right:
            merge_decisions[idx] = "left"
            if verbose:
                logger.debug(
                    "  seg[%d]: merging short-valley with left neighbor (no right neighbor)",
                    idx,
                )

        elif has_right and not has_left:
            merge_decisions[idx] = "right"
            if verbose:
                logger.debug(
                    "  seg[%d]: merging short-valley with right neighbor (no left neighbor)",
                    idx,
                )

        else:
            # Both neighbors exist — choose the better merge target
            left_seg = segments[left_idx]
            right_seg = segments[right_idx]
            current_seg = segments[idx]

            left_gap = float(current_seg["start"]) - float(left_seg["end"])
            right_gap = float(right_seg["start"]) - float(current_seg["end"])
            left_dur = float(left_seg.get("duration", 0.0))
            right_dur = float(right_seg.get("duration", 0.0))

            if left_gap < 0:
                decision = "left"
                reason = "overlaps with left"
            elif right_gap < 0:
                decision = "right"
                reason = "overlaps with right"
            elif left_gap <= right_gap * 0.5:
                decision = "left"
                reason = f"left gap ({left_gap:.3f}s) << right gap ({right_gap:.3f}s)"
            elif right_gap <= left_gap * 0.5:
                decision = "right"
                reason = f"right gap ({right_gap:.3f}s) << left gap ({left_gap:.3f}s)"
            elif left_dur <= right_dur:
                decision = "left"
                reason = f"left neighbor shorter ({left_dur:.3f}s ≤ {right_dur:.3f}s)"
            else:
                decision = "right"
                reason = f"right neighbor shorter ({right_dur:.3f}s < {left_dur:.3f}s)"

            merge_decisions[idx] = decision
            if verbose:
                logger.debug(
                    "  seg[%d]: merging short-valley %s (%s) - gaps: L=%.3fs, R=%.3fs, durs: L=%.3fs, R=%.3fs",
                    idx,
                    decision,
                    reason,
                    left_gap,
                    right_gap,
                    left_dur,
                    right_dur,
                )

    if not merge_decisions:
        return segments

    # ── Step 3: Build the set of absorbed indices ──────────────────────────
    # An "absorbed" index is any segment that will be part of a merge group
    # and should NOT be emitted independently.
    #
    # This includes:
    #   - Short-valley segments with a merge decision (they merge into a neighbor)
    #   - The TARGET neighbors that short-valley segments merge into
    #
    # Building this set BEFORE emission prevents the bug where a left neighbor
    # is emitted independently AND as part of a merged group (duplicate segments
    # with negative gaps).
    absorbed = set()

    # Mark all short-valley segments with merge decisions
    for idx in merge_decisions:
        absorbed.add(idx)

    # Mark the targets of "left" merges (the segment to the left that absorbs
    # the short-valley segment)
    for idx in sorted(merge_decisions.keys()):
        if merge_decisions[idx] == "left":
            target = idx - 1
            # Skip past any consecutive merge segments to find the real anchor
            while target >= 0 and target in merge_decisions:
                target -= 1
            if target >= 0:
                absorbed.add(target)

    # Mark the targets of "right" merges (the segment to the right)
    for idx in sorted(merge_decisions.keys()):
        if merge_decisions[idx] == "right":
            target = idx + 1
            while target < len(segments) and target in merge_decisions:
                target += 1
            if target < len(segments):
                absorbed.add(target)

    if verbose:
        logger.debug(
            "  merge_decisions=%s, absorbed=%s",
            {k: v for k, v in sorted(merge_decisions.items())},
            sorted(absorbed),
        )

    # ── Step 4: Emit segments, merging groups as we go ─────────────────────
    merged_segments = []
    i = 0
    while i < len(segments):
        if i not in absorbed:
            # Not part of any merge — emit as-is
            merged_segments.append(segments[i])
            i += 1
            continue

        # This segment is part of a merge group.
        # Find the full contiguous extent of this group.

        # Expand left to find the anchor (the first non-absorbed segment
        # that is the ultimate target of "right" merges from the left)
        group_start = i
        while group_start > 0:
            prev = group_start - 1
            if prev in absorbed:
                # Check if prev is a short-valley that merges right into group_start
                if prev in merge_decisions and merge_decisions[prev] == "right":
                    group_start = prev
                else:
                    break
            else:
                break

        # Expand right to find the final target
        group_end = i
        while group_end < len(segments):
            # If group_end is a short-valley merging right, include its target
            if group_end in merge_decisions and merge_decisions[group_end] == "right":
                target = group_end + 1
                while target < len(segments) and target in merge_decisions:
                    target += 1
                if target < len(segments):
                    group_end = target
                else:
                    break
            # If group_end+1 merges left into group_end, include group_end+1
            elif (
                group_end + 1 < len(segments)
                and group_end + 1 in merge_decisions
                and merge_decisions[group_end + 1] == "left"
            ):
                group_end = group_end + 1
            else:
                break

        # Build merged segment from [group_start, group_end] inclusive
        merge_indices = list(range(group_start, group_end + 1))
        merge_group = [segments[j] for j in merge_indices]

        first_seg = merge_group[0]
        last_seg = merge_group[-1]

        merged_seg = copy.deepcopy(first_seg)
        merged_seg["start"] = first_seg["start"]
        merged_seg["end"] = last_seg["end"]
        merged_seg["duration"] = float(last_seg["end"]) - float(first_seg["start"])
        merged_seg["end_reason"] = last_seg.get("end_reason")

        all_probs = []
        for seg in merge_group:
            all_probs.extend(seg.get("segment_probs", []))
        merged_seg["segment_probs"] = all_probs
        if all_probs:
            merged_seg["prob"] = float(np.mean(all_probs))
            merged_seg["frames_length"] = len(all_probs)
        else:
            merged_seg["prob"] = 0.0
            merged_seg["frames_length"] = 0

        merged_seg["had_overflow"] = had_overflow or any(
            seg.get("had_overflow", False) for seg in merge_group
        )

        if verbose:
            logger.debug(
                "  MERGED group indices=%s: [%.3fs → %.3fs] dur=%.3fs "
                "(merged %d segments, end_reason=%s)",
                merge_indices,
                float(merged_seg["start"]),
                float(merged_seg["end"]),
                float(merged_seg["duration"]),
                len(merge_group),
                merged_seg.get("end_reason"),
            )

        merged_segments.append(merged_seg)
        i = group_end + 1

    if verbose:
        logger.debug(
            "_merge_short_valley_segments: %d segments → %d segments after merging",
            len(segments),
            len(merged_segments),
        )

    # ── Step 5: Recursively check for remaining short valleys ──────────────
    if len(merged_segments) < len(segments):
        return _merge_short_valley_segments(
            merged_segments,
            min_duration_sec,
            audio_np,
            sample_rate,
            original_start_sec,
            had_overflow,
            orig_start_dt,
            probs,
            hop_sec,
            verbose,
        )

    return merged_segments


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
    min_sub_segment_duration_sec: float = DEFAULT_MIN_SUB_SEG_DURATION_SEC,
    verbose: bool = False,
) -> List[SpeechSegment]:
    """
    Run FireRedVAD on a speech segment's audio and split it into independent
    sub-segments. Each sub-segment has properly adjusted global timestamps
    and its own per-frame ``segment_probs`` sliced from the VAD output.

    **Audio Boundary Fix:**
    If the audio numpy array duration differs from the declared segment
    duration (indicating the VAD worker's overlap trimming shifted the
    audio bounds), the segment metadata is corrected to match the actual
    audio. Sub-segment boundaries are also clamped to prevent negative
    or out-of-bounds offsets.

    **Minimum Duration Enforcement (Valley-Split Only):**
    Sub-segments shorter than ``min_sub_segment_duration_sec`` are intelligently
    merged back into adjacent segments, **but only when the sub-segment's
    ``end_reason`` is ``"valley"``**. This ensures the merge logic only applies
    to segments split at natural speech pauses, and does not conflict with the
    ``ShortSegmentAccumulator`` which handles segments ending due to
    ``hard_limit``, ``silence``, or other reasons.

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
        Minimum duration for a sub-segment to be kept. Segments shorter than this
        **AND** with ``end_reason == "valley"`` will be merged with adjacent segments.
        Non-valley short segments are left for ``ShortSegmentAccumulator``.
        If the entire parent segment is shorter than this threshold, splitting is
        skipped entirely to avoid redundant VAD work.

    Returns
    -------
    List[SpeechSegment]
        List of independent sub-segments with:
        - adjusted global timestamps
        - per-frame ``segment_probs`` (sliced from the VAD output)
        Returns a single-element list with the original segment if:
        - Audio is empty
        - Segment is too short to split (see guards below)
        - VAD fails
        - VAD finds no sub-segments
        - All sub-segments are filtered out or merged back to a single segment
    """
    if audio_np is None or audio_np.size == 0:
        logger.warning(
            "split_segment_with_vad: empty audio for segment %s — returning as-is",
            segment.get("num"),
        )
        return [segment]

    # ── FIX 1: Correct segment metadata to match actual audio ──────────────
    # The VAD worker trims the buffer with overlap (trim_overlap_sec=0.300s),
    # which can cause the emitted segment's declared start/end/duration to
    # not match the actual audio slice. The audio is the source of truth.
    #
    # Example: VAD worker emits segment [0.000, 5.640] dur=5.640s, but
    # the actual audio slice starts at buffer position 0.300s, so the
    # audio array is only 5.340s long.
    #
    # Instead of trimming the audio (which would lose content), we correct
    # the segment metadata so all downstream calculations are consistent.
    declared_duration = float(segment.get("duration", 0.0))
    actual_duration = len(audio_np) / sample_rate
    duration_mismatch = actual_duration - declared_duration

    if abs(duration_mismatch) > 0.05:  # More than 50ms mismatch
        logger.warning(
            "split_segment_with_vad: segment %s declared duration %.3fs but "
            "audio is %.3fs (diff=%+.3fs). Correcting segment metadata to match audio. "
            "This typically happens when the VAD worker trims the buffer with overlap.",
            segment.get("num"),
            declared_duration,
            actual_duration,
            duration_mismatch,
        )
        # Fix the segment's duration and end time to match the actual audio
        segment["duration"] = actual_duration
        segment["end"] = float(segment["start"]) + actual_duration
        if verbose:
            logger.debug(
                "  corrected segment: start=%.3fs, end=%.3fs, duration=%.3fs",
                float(segment["start"]),
                float(segment["end"]),
                actual_duration,
            )

    # Use corrected duration going forward
    duration_sec = actual_duration

    # ── Existing guards ─────────────────────────────────────────────────────
    if duration_sec < min_speech_duration_sec * 2:
        logger.debug(
            "split_segment_with_vad: segment %s too short (%.3fs < %.3fs) "
            "for splitting — returning as-is",
            segment.get("num"),
            duration_sec,
            min_speech_duration_sec * 2,
        )
        return [segment]

    if min_sub_segment_duration_sec > 0 and duration_sec < min_sub_segment_duration_sec:
        logger.info(
            "split_segment_with_vad: segment %s (%.3fs) is shorter than "
            "min_sub_segment_duration_sec (%.3fs) — splitting would be "
            "immediately merged back; skipping VAD and returning as-is",
            segment.get("num"),
            duration_sec,
            min_sub_segment_duration_sec,
        )
        return [segment]

    logger.info(
        "split_segment_with_vad: running secondary VAD on segment %s (%.3fs, %d samples)",
        segment.get("num"),
        duration_sec,
        len(audio_np),
    )

    audio_np, _ = normalize_audio_for_vad(audio_np, sample_rate)
    duration = get_audio_duration(audio_np, sample_rate)
    if duration >= DEFAULT_SOFT_LIMIT_SEC:
        audio_np, _ = quantize_audio(
            audio_np,
            target_dtype="float16",
            sr=sample_rate,
            verbose=verbose,
        )

    # ── Run VAD ─────────────────────────────────────────────────────────────
    try:
        sub_segments, probs = extract_speech_timestamps(
            audio=audio_np,
            threshold=threshold,
            min_silence_duration_sec=min_silence_duration_sec,
            min_speech_duration_sec=min_speech_duration_sec,
            max_speech_duration_sec=max_speech_duration_sec,
            return_seconds=True,
            with_scores=True,
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

    # ── Build absolute sub-segments ─────────────────────────────────────────
    # Use the CORRECTED segment start (matches actual audio bounds)
    original_start_sec = float(segment["start"])
    original_end_sec = float(segment["end"])
    original_start_utc = segment.get("start_time_utc")
    had_overflow = segment.get("had_overflow", False)

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

    hop_sec = 0.010
    # Use the actual audio length as the maximum sub-segment boundary
    max_sub_end = len(audio_np) / sample_rate

    absolute_sub_segments: List[SpeechSegment] = []
    for sub_seg in sub_segments:
        sub_start_sec = float(sub_seg["start"])
        sub_end_sec = float(sub_seg["end"])

        # ── FIX 2: Clamp sub-segment boundaries to valid range ──────────────
        # VAD can produce boundaries outside the audio range due to
        # frame-level rounding or pre/post-roll adjustments in the
        # secondary VAD pass. Clamp to the actual audio bounds.
        sub_start_sec = max(0.0, min(sub_start_sec, max_sub_end))
        sub_end_sec = max(sub_start_sec, min(sub_end_sec, max_sub_end))

        sub_duration = sub_end_sec - sub_start_sec

        if sub_duration <= 0.001:  # Effectively zero or negative
            logger.warning(
                "split_segment_with_vad: sub-segment has non-positive duration "
                "[%.3fs, %.3fs] after clamping (dur=%.3fs) — skipping",
                sub_start_sec,
                sub_end_sec,
                sub_duration,
            )
            continue

        start_sample = int(round(sub_start_sec * sample_rate))
        end_sample = int(round(sub_end_sec * sample_rate))

        # ── FIX 3: Guard against out-of-bounds sample indices ───────────────
        start_sample = max(0, min(start_sample, len(audio_np)))
        end_sample = max(start_sample + 1, min(end_sample, len(audio_np)))

        sub_audio = audio_np[start_sample:end_sample].copy()

        if sub_audio.size == 0:
            logger.warning(
                "split_segment_with_vad: sub-segment [%.3fs, %.3fs] of segment %s "
                "has empty audio after sample extraction — skipping",
                sub_start_sec,
                sub_end_sec,
                segment.get("num"),
            )
            continue

        frame_start_idx = int(sub_start_sec / hop_sec)
        frame_end_idx = int(sub_end_sec / hop_sec)
        sub_probs = probs[frame_start_idx : frame_end_idx + 1] if probs else []

        new_seg = copy.deepcopy(segment)
        # Global timestamps: offset from the CORRECTED segment start
        new_seg["start"] = original_start_sec + sub_start_sec
        new_seg["end"] = original_start_sec + sub_end_sec
        new_seg["duration"] = sub_duration
        new_seg.pop("num", None)
        new_seg["segment_probs"] = sub_probs
        new_seg["had_overflow"] = had_overflow

        if sub_probs:
            new_seg["prob"] = float(np.mean(sub_probs))
            new_seg["frames_length"] = len(sub_probs)
            new_seg["frame_start"] = frame_start_idx
            new_seg["frame_end"] = frame_end_idx

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
                "(dur=%.3fs, %d samples, %d probs, end_reason=%s)",
                float(new_seg["start"]),
                float(new_seg["end"]),
                sub_duration,
                len(sub_audio),
                len(sub_probs),
                new_seg.get("end_reason"),
            )

        absolute_sub_segments.append(new_seg)

    if not absolute_sub_segments:
        logger.warning(
            "split_segment_with_vad: all sub-segments had empty audio — "
            "returning original segment %s",
            segment.get("num"),
        )
        return [segment]

    # ── Valley-segment merging ──────────────────────────────────────────────
    if min_sub_segment_duration_sec > 0 and len(absolute_sub_segments) > 1:
        before_merge = len(absolute_sub_segments)
        if verbose:
            end_reasons = [s.get("end_reason") for s in absolute_sub_segments]
            logger.debug(
                "split_segment_with_vad: end_reasons before merge: %s",
                {reason: end_reasons.count(reason) for reason in set(end_reasons)},
            )
        absolute_sub_segments = _merge_short_valley_segments(
            segments=absolute_sub_segments,
            min_duration_sec=min_sub_segment_duration_sec,
            audio_np=audio_np,
            sample_rate=sample_rate,
            original_start_sec=original_start_sec,
            had_overflow=had_overflow,
            orig_start_dt=orig_start_dt,
            probs=probs,
            hop_sec=hop_sec,
            verbose=verbose,
        )
        if len(absolute_sub_segments) < before_merge:
            logger.info(
                "split_segment_with_vad: merged %d short-valley sub-segments into %d segments "
                "(min_duration=%.3fs, %d non-valley short segments left for accumulator)",
                before_merge - len(absolute_sub_segments),
                len(absolute_sub_segments),
                min_sub_segment_duration_sec,
                sum(
                    1
                    for s in absolute_sub_segments
                    if float(s.get("duration", 0.0)) < min_sub_segment_duration_sec
                ),
            )

    # ── Final guard: single segment covering full audio → return original ────
    if len(absolute_sub_segments) == 1:
        only_seg = absolute_sub_segments[0]
        # Check if the single sub-segment covers essentially the entire audio
        coverage_start = float(only_seg["start"]) - original_start_sec
        coverage_end = float(only_seg["end"]) - original_start_sec
        if abs(coverage_start) < 0.05 and abs(coverage_end - duration_sec) < 0.05:
            logger.info(
                "split_segment_with_vad: after merging, single sub-segment covers entire audio "
                "in segment %s — returning as-is",
                segment.get("num"),
            )
            return [segment]

    logger.info(
        "split_segment_with_vad: segment %s split into %d sub-segments "
        "(from %d initial VAD segments, min_valley_duration=%.3fs)",
        segment.get("num"),
        len(absolute_sub_segments),
        len(sub_segments),
        min_sub_segment_duration_sec,
    )

    return absolute_sub_segments
