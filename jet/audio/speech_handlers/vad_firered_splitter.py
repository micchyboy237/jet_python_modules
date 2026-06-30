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

V2 Update: Uses get_valid_speech_waves for more accurate speech wave detection
with shape analysis (prominence, excursion, etc.) instead of simple threshold-based
segmentation. Now uses get_valid_speech_waves(with_scores=True) to obtain both
valid waves and probability scores in a single call, eliminating the redundant
extract_speech_timestamps invocation.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_MAX_SPEECH_SEC,
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_MIN_SPEECH_SEC,
    DEFAULT_MIN_SUB_SEG_DURATION_SEC,
    DEFAULT_THRESHOLD,
    DEFAULT_USE_HYBRID,
)
from jet.audio.audio_waveform.vad.vad_firered import (
    DEFAULT_MAX_BUFFER_SEC,
    DEFAULT_PAD_START_FRAME,
    DEFAULT_SMOOTH_WINDOW_SIZE,
)
from jet.audio.helpers.config import HOP_SIZE
from jet.audio.speech.firered.speech_waves import (
    DEFAULT_BASELINE_THRESHOLD,
    DEFAULT_MIN_DURATION_SEC,
    DEFAULT_MIN_EXCURSION,
    DEFAULT_MIN_FRAMES,
    DEFAULT_MIN_PEAK_PROB,
    DEFAULT_MIN_PROMINENCE,
    get_valid_speech_waves,
)
from rich.console import Console

console = Console()


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
        console.print(
            "[blue]_should_consider_for_merging:[/blue] "
            "segment [%.3fs → %.3fs] dur=%.3fs is short but end_reason=%s (not 'valley') — skipping merge, "
            "delegating to ShortSegmentAccumulator"
            % (
                float(segment["start"]),
                float(segment["end"]),
                duration,
                end_reason,
            ),
            style="dim",
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
            console.print(
                "[cyan]_merge_short_valley_segments:[/cyan] %d segments, %d short (< %.3fs), "
                "but none are valley-split — nothing to merge"
                % (len(segments), total_short, min_duration_sec),
                style="dim",
            )
        return segments

    if verbose:
        total_short = sum(
            1 for seg in segments if float(seg.get("duration", 0.0)) < min_duration_sec
        )
        console.print(
            "[cyan]_merge_short_valley_segments:[/cyan] %d segments total, %d short (< %.3fs), %d short-valley (eligible for merge)"
            % (len(segments), total_short, min_duration_sec, len(short_valley_indices)),
            style="dim",
        )
        for idx in short_valley_indices:
            seg = segments[idx]
            console.print(
                "  [cyan]short-valley seg[%d]:[/cyan] [%.3fs → %.3fs] dur=%.3fs end_reason=%s"
                % (
                    idx,
                    float(seg["start"]),
                    float(seg["end"]),
                    float(seg.get("duration", 0.0)),
                    seg.get("end_reason"),
                ),
                style="dim",
            )

    merge_decisions = {}
    for idx in short_valley_indices:
        has_left = False
        left_idx = idx - 1
        while left_idx >= 0:
            if left_idx not in merge_decisions:
                has_left = True
                break
            left_idx -= 1

        has_right = False
        right_idx = idx + 1
        while right_idx < len(segments):
            if right_idx not in merge_decisions:
                has_right = True
                break
            right_idx += 1

        if not has_left and not has_right:
            if verbose:
                console.print(
                    f"  [cyan]seg[{idx}]:[/cyan] isolated short-valley segment - keeping as-is",
                    style="dim",
                )
            continue

        if has_left and not has_right:
            merge_decisions[idx] = "left"
            if verbose:
                console.print(
                    f"  [cyan]seg[{idx}]:[/cyan] merging short-valley with left neighbor (no right neighbor)",
                    style="dim",
                )
        elif has_right and not has_left:
            merge_decisions[idx] = "right"
            if verbose:
                console.print(
                    f"  [cyan]seg[{idx}]:[/cyan] merging short-valley with right neighbor (no left neighbor)",
                    style="dim",
                )
        else:
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
                console.print(
                    (
                        f"  [cyan]seg[{idx}]:[/cyan] merging short-valley {decision} ({reason}) - "
                        f"gaps: L={left_gap:.3f}s, R={right_gap:.3f}s, durs: L={left_dur:.3f}s, R={right_dur:.3f}s"
                    ),
                    style="dim",
                )

    if not merge_decisions:
        return segments

    absorbed = set()
    for idx in merge_decisions:
        absorbed.add(idx)

    for idx in sorted(merge_decisions.keys()):
        if merge_decisions[idx] == "left":
            target = idx - 1
            while target >= 0 and target in merge_decisions:
                target -= 1
            if target >= 0:
                absorbed.add(target)

    for idx in sorted(merge_decisions.keys()):
        if merge_decisions[idx] == "right":
            target = idx + 1
            while target < len(segments) and target in merge_decisions:
                target += 1
            if target < len(segments):
                absorbed.add(target)

    if verbose:
        console.print(
            "  [cyan]merge_decisions[/cyan]=%s, [cyan]absorbed[/cyan]=%s"
            % (
                {k: v for k, v in sorted(merge_decisions.items())},
                sorted(absorbed),
            ),
            style="dim",
        )

    merged_segments = []
    i = 0
    while i < len(segments):
        if i not in absorbed:
            merged_segments.append(segments[i])
            i += 1
            continue

        # Find the full merge group
        group_start = i
        while group_start > 0:
            prev = group_start - 1
            if prev in absorbed:
                if prev in merge_decisions and merge_decisions[prev] == "right":
                    group_start = prev
                else:
                    break
            else:
                break

        group_end = i
        while group_end < len(segments):
            if group_end in merge_decisions and merge_decisions[group_end] == "right":
                target = group_end + 1
                while target < len(segments) and target in merge_decisions:
                    target += 1
                if target < len(segments):
                    group_end = target
                else:
                    break
            elif (
                group_end + 1 < len(segments)
                and group_end + 1 in merge_decisions
                and merge_decisions[group_end + 1] == "left"
            ):
                group_end = group_end + 1
            else:
                break

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
            console.print(
                "  [green]MERGED group indices=%s[/green]: [%.3fs → %.3fs] dur=%.3fs (merged %d segments, end_reason=%s)"
                % (
                    merge_indices,
                    float(merged_seg["start"]),
                    float(merged_seg["end"]),
                    float(merged_seg["duration"]),
                    len(merge_group),
                    merged_seg.get("end_reason"),
                ),
                style="bold",
            )

        merged_segments.append(merged_seg)
        i = group_end + 1

    if verbose:
        console.print(
            "[cyan]_merge_short_valley_segments:[/cyan] %d segments → %d segments after merging"
            % (len(segments), len(merged_segments)),
            style="dim",
        )

    # Recursively merge if we still reduced the count
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


def _convert_wave_to_segment(
    wave: dict,
    segment: SpeechSegment,
    audio_np: np.ndarray,
    sample_rate: int,
    original_start_sec: float,
    had_overflow: bool,
    orig_start_dt: Optional[datetime],
    probs: List[float],
    hop_sec: float,
    verbose: bool = False,
) -> Optional[SpeechSegment]:
    """
    Convert a SpeechWave to a SpeechSegment with adjusted global timestamps.

    Args:
        wave: SpeechWave dictionary from get_valid_speech_waves
        segment: Original parent segment for metadata inheritance
        audio_np: Audio data for this segment
        sample_rate: Sample rate of the audio
        original_start_sec: Absolute start time of parent segment
        had_overflow: Whether parent segment had overflow
        orig_start_dt: Parsed start datetime of parent segment
        probs: Full probability array from VAD (for segment_probs slicing)
        hop_sec: Hop size in seconds
        verbose: Enable debug logging

    Returns:
        SpeechSegment with adjusted timestamps, or None if conversion fails
    """
    wave_start_sec = float(wave["start_sec"])
    wave_end_sec = float(wave["end_sec"])
    wave_duration = wave_end_sec - wave_start_sec

    # Clamp to valid audio bounds
    max_audio_sec = len(audio_np) / sample_rate
    wave_start_sec = max(0.0, min(wave_start_sec, max_audio_sec))
    wave_end_sec = max(wave_start_sec, min(wave_end_sec, max_audio_sec))
    wave_duration = wave_end_sec - wave_start_sec

    if wave_duration <= 0.001:
        console.print(
            "[yellow]_convert_wave_to_segment:[/yellow] wave has non-positive duration "
            "[%.3fs, %.3fs] — skipping" % (wave_start_sec, wave_end_sec),
            style="bold yellow",
        )
        return None

    # Extract audio samples
    start_sample = int(round(wave_start_sec * sample_rate))
    end_sample = int(round(wave_end_sec * sample_rate))
    start_sample = max(0, min(start_sample, len(audio_np)))
    end_sample = max(start_sample + 1, min(end_sample, len(audio_np)))
    sub_audio = audio_np[start_sample:end_sample].copy()

    if sub_audio.size == 0:
        console.print(
            "[yellow]_convert_wave_to_segment:[/yellow] wave [%.3fs, %.3fs] has empty audio — skipping"
            % (wave_start_sec, wave_end_sec),
            style="bold yellow",
        )
        return None

    # Create new segment from parent
    new_seg = copy.deepcopy(segment)

    # Adjust timestamps to global coordinates
    new_seg["start"] = original_start_sec + wave_start_sec
    new_seg["end"] = original_start_sec + wave_end_sec
    new_seg["duration"] = wave_duration
    new_seg.pop("num", None)

    # Set wave-specific properties
    details = wave.get("details", {})
    frame_start = details.get("frame_start", int(wave_start_sec / hop_sec))
    frame_end = details.get("frame_end", int(wave_end_sec / hop_sec))

    new_seg["frame_start"] = frame_start
    new_seg["frame_end"] = frame_end
    new_seg["frames_length"] = frame_end - frame_start

    # Slice probability scores for this wave from the full probability array
    if probs:
        sub_probs = (
            probs[frame_start : frame_end + 1] if frame_start < len(probs) else []
        )
        new_seg["segment_probs"] = sub_probs
        if sub_probs:
            new_seg["prob"] = float(np.mean(sub_probs))
            new_seg["frames_length"] = len(sub_probs)
        else:
            new_seg["prob"] = details.get("avg_prob", 0.0)
            new_seg["frames_length"] = 0
    else:
        new_seg["segment_probs"] = []
        new_seg["prob"] = details.get("avg_prob", 0.0)
        new_seg["frames_length"] = 0

    new_seg["had_overflow"] = had_overflow

    # Set end_reason based on wave characteristics
    # Waves that end due to falling below threshold are "valley" type
    new_seg["end_reason"] = "valley"  # Speech waves end at natural valleys

    # Handle UTC timestamps
    if orig_start_dt is not None:
        new_start_ts = orig_start_dt.timestamp() + wave_start_sec
        new_start_dt = datetime.fromtimestamp(new_start_ts, tz=timezone.utc)
        new_seg["start_time_utc"] = new_start_dt.isoformat()

        new_end_ts = new_start_dt.timestamp() + wave_duration
        new_end_dt = datetime.fromtimestamp(new_end_ts, tz=timezone.utc)
        new_seg["end_time_utc"] = new_end_dt.isoformat()
    else:
        new_seg.pop("start_time_utc", None)
        new_seg.pop("end_time_utc", None)

    # Add wave-specific metadata
    new_seg["wave_details"] = {
        "prominence": details.get("prominence", 0.0),
        "excursion": details.get("excursion", 0.0),
        "baseline": details.get("baseline", 0.0),
        "composite_score": details.get("composite_score", 0.0),
        "is_valid": wave.get("is_valid", True),
    }

    if verbose:
        console.print(
            "[green]_convert_wave_to_segment:[/green] created segment [%.3fs → %.3fs global] (dur=%.3fs, %d samples, avg_prob=%.3f, end_reason=valley)"
            % (
                float(new_seg["start"]),
                float(new_seg["end"]),
                wave_duration,
                len(sub_audio),
                float(new_seg.get("prob", 0.0)),
            ),
            style="bold",
        )

    return new_seg


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
    use_hybrid: bool = DEFAULT_USE_HYBRID,
    # New wave-specific parameters
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_excursion: float = DEFAULT_MIN_EXCURSION,
    min_peak_prob: float = DEFAULT_MIN_PEAK_PROB,
    min_frames: int = DEFAULT_MIN_FRAMES,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    baseline_threshold: float = DEFAULT_BASELINE_THRESHOLD,
    verbose: bool = False,
) -> List[SpeechSegment]:
    """
    Run FireRedVAD on a speech segment's audio and split it into independent
    sub-segments using speech wave detection with shape analysis.

    **V2 Update:** Now uses get_valid_speech_waves with shape validation
    (prominence, excursion, etc.) for more accurate detection of genuine
    speech waves. Probability scores are obtained via the new with_scores=True
    parameter, eliminating the need for a separate extract_speech_timestamps call.

    Each sub-segment has properly adjusted global timestamps and its own
    per-frame ``segment_probs`` sliced from the VAD output.

    **Audio Boundary Fix:**
    If the audio numpy array duration differs from the declared segment
    duration (indicating the VAD worker's overlap trimming shifted the
    audio bounds), the segment metadata is corrected to match the actual
    audio. Sub-segment boundaries are also clamped to prevent negative
    or out-of-bounds offsets.

    **Minimum Duration Enforcement (Valley-Split Only):**
    Sub-segments shorter than ``min_sub_segment_duration_sec`` are intelligently
    merged back into adjacent segments, **but only when the sub-segment's
    ``end_reason`` is ``"valley"``**. Since speech waves naturally end at valleys,
    all wave-based segments will have end_reason="valley" by default.

    Parameters
    ----------
    segment:
        The original SpeechSegment (with global start/end timestamps).
    audio_np:
        The audio data for this segment (numpy array).
    sample_rate:
        Sample rate of the audio.
    threshold, min_silence_duration_sec, min_speech_duration_sec, etc.:
        VAD parameters passed to get_valid_speech_waves.
    min_prominence, min_excursion, min_peak_prob, min_frames, min_duration_sec, baseline_threshold:
        Wave detection parameters for shape-based validation.
    min_sub_segment_duration_sec:
        Minimum duration for a sub-segment to be kept. Segments shorter than this
        **AND** with ``end_reason == "valley"`` will be merged with adjacent segments.
        If the entire parent segment is shorter than this threshold, splitting is
        skipped entirely to avoid redundant VAD work.

    Returns
    -------
    List[SpeechSegment]
        List of independent sub-segments with:
        - adjusted global timestamps
        - per-frame ``segment_probs`` (sliced from the VAD output)
        - ``wave_details`` metadata (prominence, excursion, etc.)
        Returns a single-element list with the original segment if:
        - Audio is empty
        - Segment is too short to split (see guards below)
        - VAD fails
        - No valid speech waves found
        - All sub-segments are filtered out or merged back to a single segment
    """
    if audio_np is None or audio_np.size == 0:
        console.print(
            "[yellow]split_segment_with_vad:[/yellow] empty audio for segment %s — returning as-is"
            % (segment.get("num"),),
            style="bold yellow",
        )
        return [segment]

    # Fix audio duration mismatch
    duration_sec = segment["duration"]

    # Guard: too short to split
    if duration_sec < min_speech_duration_sec * 2:
        console.print(
            "[blue]split_segment_with_vad:[/blue] segment %s too short (%.3fs < %.3fs) for splitting — returning as-is"
            % (segment.get("num"), duration_sec, min_speech_duration_sec * 2),
            style="dim",
        )
        return [segment]

    if min_sub_segment_duration_sec > 0 and duration_sec < min_sub_segment_duration_sec:
        console.print(
            "[cyan]split_segment_with_vad:[/cyan] segment %s (%.3fs) is shorter than min_sub_segment_duration_sec (%.3fs) — splitting would be immediately merged back; skipping VAD and returning as-is"
            % (segment.get("num"), duration_sec, min_sub_segment_duration_sec),
            style="dim",
        )
        return [segment]

    console.print(
        "[green]split_segment_with_vad:[/green] running speech wave detection on segment %s (%.3fs, %d samples)"
        % (segment.get("num"), duration_sec, len(audio_np)),
        style="bold",
    )

    # # Normalize audio for better VAD detection
    # audio_np, _ = normalize_audio_for_vad(audio_np, sample_rate)
    # duration = get_audio_duration(audio_np, sample_rate)
    # if duration >= DEFAULT_SOFT_LIMIT_SEC:
    #     audio_np, _ = quantize_audio(
    #         audio_np,
    #         target_dtype="float16",
    #         sr=sample_rate,
    #     )

    try:
        # Use get_valid_speech_waves with with_scores=True to get both waves and probability scores
        # This eliminates the need for a separate extract_speech_timestamps call
        result = get_valid_speech_waves(
            audio=audio_np,
            # sampling_rate=sample_rate,
            # vad_threshold=threshold,
            min_prominence=min_prominence,
            min_excursion=min_excursion,
            min_peak_prob=min_peak_prob,
            min_frames=min_frames,
            min_duration_sec=min_duration_sec,
            baseline_threshold=baseline_threshold,
            # min_speech_duration_ms=int(min_speech_duration_sec * 1000),
            # min_silence_duration_ms=int(min_silence_duration_sec * 1000),
            with_audio=False,
            with_scores=True,  # Get probability scores alongside waves
        )

        # Unpack result based on with_scores=True
        # Returns Tuple[List[SpeechWave], List[float]]
        valid_waves, probs = result

        console.print(
            "[green]split_segment_with_vad:[/green] detected %d valid speech waves in segment %s | duration=%.3fs | probs=%d"
            % (
                len(valid_waves),
                segment.get("num"),
                duration_sec,
                len(probs) if probs is not None else 0,
            ),
            style="bold",
        )

        console.print(
            f"[grey37]Valid waves:[/grey37]\n{json.dumps(valid_waves, ensure_ascii=False)}",
            style="dim",
        )

    except Exception as exc:
        console.print(
            "[red]split_segment_with_vad:[/red] speech wave detection failed for segment %s: %s — returning as-is"
            % (segment.get("num"), exc),
            style="bold red",
        )
        return [segment]

    if not valid_waves:
        console.print(
            "[bold yellow]split_segment_with_vad:[/bold yellow] no valid speech waves found in segment %s — returning as-is"
            % (segment.get("num"),),
            style="bold yellow",
        )
        return [segment]

    # Extract metadata from parent segment
    original_start_sec = float(segment["start"])
    original_start_utc = segment.get("start_time_utc")
    had_overflow = segment.get("had_overflow", False)

    # Parse UTC timestamp if available
    orig_start_dt: Optional[datetime] = None
    if original_start_utc is not None:
        try:
            orig_start_dt = datetime.fromisoformat(str(original_start_utc))
            if orig_start_dt.tzinfo is None:
                orig_start_dt = orig_start_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError) as exc:
            console.print(
                "[yellow]split_segment_with_vad:[/yellow] could not parse start_time_utc=%s: %s"
                % (original_start_utc, exc),
                style="dim",
            )

    hop_sec = HOP_SIZE / sample_rate

    # Convert waves to segments
    absolute_sub_segments: List[SpeechSegment] = []
    for wave in valid_waves:
        sub_seg = _convert_wave_to_segment(
            wave=wave,
            segment=segment,
            audio_np=audio_np,
            sample_rate=sample_rate,
            original_start_sec=original_start_sec,
            had_overflow=had_overflow,
            orig_start_dt=orig_start_dt,
            probs=probs,
            hop_sec=hop_sec,
            verbose=verbose,
        )

        if sub_seg is None:
            continue

        absolute_sub_segments.append(sub_seg)

    if not absolute_sub_segments:
        console.print(
            "[yellow]split_segment_with_vad:[/yellow] all waves converted to empty segments — returning original segment %s"
            % (segment.get("num"),),
            style="bold yellow",
        )
        return [segment]

    # Merge short valley segments (all wave segments have end_reason="valley")
    if min_sub_segment_duration_sec > 0 and len(absolute_sub_segments) > 1:
        before_merge = len(absolute_sub_segments)

        if verbose:
            end_reasons = [s.get("end_reason") for s in absolute_sub_segments]
            console.print(
                "[yellow]split_segment_with_vad:[/yellow] end_reasons before merge: %s"
                % ({reason: end_reasons.count(reason) for reason in set(end_reasons)}),
                style="dim",
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
            console.print(
                "[green]split_segment_with_vad:[/green] merged %d short-valley sub-segments into %d segments (min_duration=%.3fs, %d non-valley short segments left for accumulator)"
                % (
                    before_merge - len(absolute_sub_segments),
                    len(absolute_sub_segments),
                    min_sub_segment_duration_sec,
                    sum(
                        1
                        for s in absolute_sub_segments
                        if float(s.get("duration", 0.0)) < min_sub_segment_duration_sec
                    ),
                ),
                style="green",
            )

    # Check if merging reduced to single segment covering full audio
    if len(absolute_sub_segments) == 1:
        only_seg = absolute_sub_segments[0]
        coverage_start = float(only_seg["start"]) - original_start_sec
        coverage_end = float(only_seg["end"]) - original_start_sec

        if abs(coverage_start) < 0.05 and abs(coverage_end - duration_sec) < 0.05:
            console.print(
                "[green]split_segment_with_vad:[/green] after merging, single sub-segment covers entire audio in segment %s — returning as-is"
                % (segment.get("num"),),
                style="dim",
            )
            return [segment]

    console.print(
        "[bold green]split_segment_with_vad:[/bold green] segment %s split into %d sub-segments (from %d valid speech waves, min_valley_duration=%.3fs)"
        % (
            segment.get("num"),
            len(absolute_sub_segments),
            len(valid_waves),
            min_sub_segment_duration_sec,
        ),
    )

    return absolute_sub_segments


# Backward compatibility aliases
split_segment_with_vad_v1 = split_segment_with_vad  # Original function name


def split_segment_with_vad_v2(
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
    use_hybrid: bool = DEFAULT_USE_HYBRID,
    # Wave-specific parameters with defaults
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_excursion: float = DEFAULT_MIN_EXCURSION,
    min_peak_prob: float = DEFAULT_MIN_PEAK_PROB,
    min_frames: int = DEFAULT_MIN_FRAMES,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    baseline_threshold: float = DEFAULT_BASELINE_THRESHOLD,
    verbose: bool = False,
) -> List[SpeechSegment]:
    """
    Version 2 of split_segment_with_vad that explicitly uses speech wave detection.
    This is an alias for backward compatibility - use split_segment_with_vad()
    which now defaults to wave-based detection with optimized score retrieval.

    All parameters are identical to split_segment_with_vad().
    """
    return split_segment_with_vad(
        segment=segment,
        audio_np=audio_np,
        sample_rate=sample_rate,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
        smooth_window_size=smooth_window_size,
        pad_start_frame=pad_start_frame,
        max_buffer_sec=max_buffer_sec,
        min_sub_segment_duration_sec=min_sub_segment_duration_sec,
        use_hybrid=use_hybrid,
        min_prominence=min_prominence,
        min_excursion=min_excursion,
        min_peak_prob=min_peak_prob,
        min_frames=min_frames,
        min_duration_sec=min_duration_sec,
        baseline_threshold=baseline_threshold,
        verbose=verbose,
    )
