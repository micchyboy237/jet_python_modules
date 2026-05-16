from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechEndReason, SpeechSegment
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_HARD_LIMIT_MIN_TROUGH_OFFSET_S,
    DEFAULT_HARD_LIMIT_MIN_VALLEY_DURATION_S,
    DEFAULT_HARD_LIMIT_SEC,
    DEFAULT_HARD_LIMIT_SMOOTHING_WINDOW,
    DEFAULT_HARD_LIMIT_TROUGH_PROMINENCE,
    DEFAULT_PROB_WEIGHT,
    DEFAULT_RETURN_SECONDS,
    DEFAULT_RMS_WEIGHT,
    DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
    DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
    DEFAULT_SOFT_LIMIT_SEC,
    DEFAULT_SOFT_LIMIT_SEC_HIGH,
    DEFAULT_SOFT_LIMIT_SEC_HIGH_MIN_TROUGH_OFFSET_S,
    DEFAULT_SOFT_LIMIT_SEC_HIGH_MIN_VALLEY_DURATION_S,
    DEFAULT_SOFT_LIMIT_SEC_HIGH_SMOOTHING_WINDOW,
    DEFAULT_SOFT_LIMIT_SEC_HIGH_TROUGH_PROMINENCE,
    DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
    DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
    DEFAULT_WITH_SCORES,
)
from jet.audio.audio_waveform.vad.vad_utils import compute_hybrid_probs
from jet.audio.helpers.config import (
    HOP_SIZE,
    HOP_STEP_S,
    SAMPLE_RATE,
)
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Pre-roll computation helper
# ---------------------------------------------------------------------------


def compute_preroll(
    onset_sample: int,
    audio_np: np.ndarray,
    probs: list[float],
    sample_rate: int,
    max_preroll_sec: float,
    hybrid_threshold: float,
    prob_weight: float,
    rms_weight: float,
    compute_hybrid: bool = False,  # ← New parameter
) -> int:
    """
    Given a speech-segment onset (in samples), look backward through the
    pre-speech audio and find how many additional samples to prepend.

    Args:
        ...
        compute_hybrid: If True, compute hybrid scores (prob + RMS).
                        If False, treat `probs` as already-computed hybrid scores.
    """
    max_preroll_samples = int(max_preroll_sec * sample_rate)

    start_sample = max(0, onset_sample - max_preroll_samples)
    lookback_audio = audio_np[start_sample:onset_sample]

    if len(lookback_audio) == 0:
        return 0

    # Align to frame grid
    n_frames = len(lookback_audio) // HOP_SIZE
    if n_frames == 0:
        return 0

    # Trim to frame-aligned length
    lookback_audio = lookback_audio[: n_frames * HOP_SIZE]

    # Extract probs slice aligned to lookback frames
    onset_frame = int(onset_sample / sample_rate / HOP_STEP_S)
    look_start_frame = onset_frame - n_frames

    lookback_probs = np.array(
        [
            probs[look_start_frame + i]
            if 0 <= look_start_frame + i < len(probs)
            else 0.0
            for i in range(n_frames)
        ],
        dtype=np.float32,
    )

    # === Hybrid handling ===
    if compute_hybrid:
        hybrid = compute_hybrid_probs(
            probs=lookback_probs,
            audio_np=lookback_audio,
            prob_weight=prob_weight,
            rms_weight=rms_weight,
            frame_samples=HOP_SIZE,
        )
    else:
        hybrid = lookback_probs  # Already hybrid scores

    # Walk backward from the frame immediately before onset
    keep_frames = 0
    for i in range(n_frames - 1, -1, -1):
        if hybrid[i] >= hybrid_threshold:
            keep_frames = n_frames - i
        else:
            break

    return keep_frames * HOP_SIZE


# ---------------------------------------------------------------------------
# Post-roll computation helper  (symmetric tail extension)
# ---------------------------------------------------------------------------


def compute_postroll(
    end_sample: int,
    audio_np: np.ndarray,
    probs: list[float],
    sample_rate: int,
    max_postroll_sec: float,
    hybrid_threshold: float,
    prob_weight: float,
    rms_weight: float,
    compute_hybrid: bool = False,  # ← New parameter
) -> int:
    """
    Given a speech-segment end (in samples), look forward through the
    post-speech audio and find how many additional samples to append.

    Args:
        ...
        compute_hybrid: If True, compute hybrid scores (prob + RMS).
                        If False, treat `probs` as already-computed hybrid scores.
    """
    max_postroll_samples = int(max_postroll_sec * sample_rate)

    stop_sample = min(len(audio_np), end_sample + max_postroll_samples)
    lookahead_audio = audio_np[end_sample:stop_sample]

    if len(lookahead_audio) == 0:
        return 0

    # Align to frame grid
    n_frames = len(lookahead_audio) // HOP_SIZE
    if n_frames == 0:
        return 0

    lookahead_audio = lookahead_audio[: n_frames * HOP_SIZE]

    # Extract probs slice aligned to lookahead frames
    end_frame = int(end_sample / sample_rate / HOP_STEP_S)
    lookahead_probs = np.array(
        [
            probs[end_frame + i] if 0 <= end_frame + i < len(probs) else 0.0
            for i in range(n_frames)
        ],
        dtype=np.float32,
    )

    # === Hybrid handling ===
    if compute_hybrid:
        hybrid = compute_hybrid_probs(
            probs=lookahead_probs,
            audio_np=lookahead_audio,
            prob_weight=prob_weight,
            rms_weight=rms_weight,
            frame_samples=HOP_SIZE,
        )
    else:
        hybrid = lookahead_probs  # Already hybrid scores

    # Walk forward from the frame immediately after the detected end
    keep_frames = 0
    for i in range(n_frames):
        if hybrid[i] >= hybrid_threshold:
            keep_frames = i + 1
        else:
            break

    return keep_frames * HOP_SIZE


@dataclass
class LimitConfig:
    """Configuration for a single limit-splitting level."""

    max_limit_sec: float
    min_valley_duration_s: float
    smoothing_window: int
    trough_prominence: float
    min_trough_offset_s: float
    end_reason: SpeechEndReason
    label: str  # For logging purposes


# Define the multi-level configuration
def _get_limit_configs() -> List[LimitConfig]:
    """Return the three-level limit configuration in order of application."""
    return [
        LimitConfig(
            max_limit_sec=DEFAULT_SOFT_LIMIT_SEC,
            min_valley_duration_s=DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
            smoothing_window=DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
            trough_prominence=DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
            min_trough_offset_s=DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
            end_reason="valley",
            label="Soft limit",
        ),
        LimitConfig(
            max_limit_sec=DEFAULT_SOFT_LIMIT_SEC_HIGH,
            min_valley_duration_s=DEFAULT_SOFT_LIMIT_SEC_HIGH_MIN_VALLEY_DURATION_S,
            smoothing_window=DEFAULT_SOFT_LIMIT_SEC_HIGH_SMOOTHING_WINDOW,
            trough_prominence=DEFAULT_SOFT_LIMIT_SEC_HIGH_TROUGH_PROMINENCE,
            min_trough_offset_s=DEFAULT_SOFT_LIMIT_SEC_HIGH_MIN_TROUGH_OFFSET_S,
            end_reason="valley",
            label="Soft limit (high)",
        ),
        LimitConfig(
            max_limit_sec=DEFAULT_HARD_LIMIT_SEC,
            min_valley_duration_s=DEFAULT_HARD_LIMIT_MIN_VALLEY_DURATION_S,
            smoothing_window=DEFAULT_HARD_LIMIT_SMOOTHING_WINDOW,
            trough_prominence=DEFAULT_HARD_LIMIT_TROUGH_PROMINENCE,
            min_trough_offset_s=DEFAULT_HARD_LIMIT_MIN_TROUGH_OFFSET_S,
            end_reason="hard_limit",
            label="Hard limit",
        ),
    ]


def apply_limit_splits(
    segments: List[SpeechSegment],
    probs: List[float],
    audio_np: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    hop_sec: float = HOP_STEP_S,
    # Keep these for backward compatibility, but they'll be overridden by the multi-level config
    max_limit_sec: Optional[float] = None,
    min_valley_duration_s: Optional[float] = None,
    smoothing_window: Optional[int] = None,
    trough_prominence: Optional[float] = None,
    min_trough_offset_s: Optional[float] = None,
    return_seconds: bool = DEFAULT_RETURN_SECONDS,
    with_scores: bool = DEFAULT_WITH_SCORES,
    end_reason_on_split: Optional[SpeechEndReason] = None,
    hybrid_prob_weight: float = DEFAULT_PROB_WEIGHT,
    hybrid_rms_weight: float = DEFAULT_RMS_WEIGHT,
) -> List[SpeechSegment]:
    """
    Apply multi-level limit splitting to speech segments.

    Applies three levels of splitting with increasingly aggressive parameters:
    1. Soft limit (DEFAULT_SOFT_LIMIT_SEC) - Conservative valley detection
    2. Soft limit high (DEFAULT_SOFT_LIMIT_SEC_HIGH) - More aggressive detection
    3. Hard limit (DEFAULT_HARD_LIMIT_SEC) - Most aggressive detection

    Backward compatibility: If any legacy parameters are provided, falls back
    to single-level splitting with those parameters.
    """

    # Check if legacy parameters were provided
    using_legacy_params = any(
        [
            max_limit_sec is not None,
            min_valley_duration_s is not None,
            smoothing_window is not None,
            trough_prominence is not None,
            min_trough_offset_s is not None,
            end_reason_on_split is not None,
        ]
    )

    if using_legacy_params:
        # Fallback to single-level splitting for backward compatibility
        console.print("[yellow]Using legacy single-level limit splitting[/yellow]")
        return _apply_single_limit_split(
            segments=segments,
            probs=probs,
            audio_np=audio_np,
            sample_rate=sample_rate,
            hop_sec=hop_sec,
            max_limit_sec=max_limit_sec or DEFAULT_SOFT_LIMIT_SEC,
            min_valley_duration_s=min_valley_duration_s
            or DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
            smoothing_window=smoothing_window or DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
            trough_prominence=trough_prominence or DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
            min_trough_offset_s=min_trough_offset_s
            or DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
            return_seconds=return_seconds,
            with_scores=with_scores,
            end_reason_on_split=end_reason_on_split or "valley",
            hybrid_prob_weight=hybrid_prob_weight,
            hybrid_rms_weight=hybrid_rms_weight,
        )

    # Multi-level splitting
    limit_configs = _get_limit_configs()

    result = segments
    for level, config in enumerate(limit_configs, 1):
        result = _apply_single_limit_split(
            segments=result,
            probs=probs,
            audio_np=audio_np,
            sample_rate=sample_rate,
            hop_sec=hop_sec,
            max_limit_sec=config.max_limit_sec,
            min_valley_duration_s=config.min_valley_duration_s,
            smoothing_window=config.smoothing_window,
            trough_prominence=config.trough_prominence,
            min_trough_offset_s=config.min_trough_offset_s,
            return_seconds=return_seconds,
            with_scores=with_scores,
            end_reason_on_split=config.end_reason,
            hybrid_prob_weight=hybrid_prob_weight,
            hybrid_rms_weight=hybrid_rms_weight,
            level_label=config.label,
        )

    return result


def _apply_single_limit_split(
    segments: List[SpeechSegment],
    probs: List[float],
    audio_np: np.ndarray,
    sample_rate: int,
    hop_sec: float,
    max_limit_sec: float,
    min_valley_duration_s: float,
    smoothing_window: int,
    trough_prominence: float,
    min_trough_offset_s: float,
    return_seconds: bool,
    with_scores: bool,
    end_reason_on_split: SpeechEndReason,
    hybrid_prob_weight: float,
    hybrid_rms_weight: float,
    level_label: str = "Limit",
) -> List[SpeechSegment]:
    """
    Apply a single level of limit-based splitting to speech segments.

    This is the extracted core logic that was previously in apply_limit_splits.
    It can be called multiple times with different configurations for multi-level
    splitting, or once for backward compatibility.
    """
    from jet.audio.speech.vad_extractors import get_best_valley_trough

    result: List[SpeechSegment] = []
    seg_num = 1

    def _split_recursive(seg: SpeechSegment) -> List[SpeechSegment]:
        """Return one or more segments produced from *seg*, splitting if needed."""
        duration = seg["duration"]
        if duration <= max_limit_sec:
            return [seg]

        frame_start: int = seg["frame_start"]
        frame_end: int = seg["frame_end"]
        seg_audio = audio_np[int(frame_start * 160) : int((frame_end + 1) * 160)]
        segment_probs = np.array(probs[frame_start : frame_end + 1], dtype=np.float32)

        seg_probs = compute_hybrid_probs(
            probs=segment_probs,
            audio_np=seg_audio,
            prob_weight=hybrid_prob_weight,
            rms_weight=hybrid_rms_weight,
        ).tolist()

        if not seg_probs:
            return [seg]

        best_trough = get_best_valley_trough(
            probs_or_audio=seg_probs,
            smoothing_window=smoothing_window,
            trough_prominence=trough_prominence,
            min_valley_duration_s=min_valley_duration_s,
            min_trough_offset_s=min_trough_offset_s,
        )

        if best_trough is None:
            console.print(
                f"[yellow]{level_label}: segment {seg['num']} is {duration:.1f}s "
                f"(>{max_limit_sec}s) but no valley trough found — keeping intact.[/yellow]"
            )
            return [seg]

        local_trough_frame: int = best_trough["frame"]
        global_trough_frame: int = frame_start + local_trough_frame
        split_time_s: float = global_trough_frame * hop_sec

        def _make_child(
            child_frame_start: int,
            child_frame_end: int,
        ) -> SpeechSegment:
            child_start_s = child_frame_start * hop_sec
            child_end_s = child_frame_end * hop_sec
            child_probs_slice = probs[child_frame_start : child_frame_end + 1]
            avg_prob = float(np.mean(child_probs_slice)) if child_probs_slice else 0.0
            duration_s = child_end_s - child_start_s
            start_val = (
                child_start_s if return_seconds else int(child_start_s * sample_rate)
            )
            end_val = child_end_s if return_seconds else int(child_end_s * sample_rate)
            return SpeechSegment(
                num=0,
                start=start_val,
                end=end_val,
                duration=duration_s,
                end_reason=None,
                prob=avg_prob,
                frames_length=len(child_probs_slice),
                frame_start=child_frame_start,
                frame_end=child_frame_end,
                type=seg["type"],
                segment_probs=child_probs_slice if with_scores else [],
            )

        left = _make_child(frame_start, global_trough_frame)
        right = _make_child(global_trough_frame, frame_end)

        console.print(
            f"[cyan]{level_label}: split segment {seg['num']} "
            f"({duration:.1f}s) at {split_time_s:.2f}s "
            f"→ {left['duration']:.1f}s + {right['duration']:.1f}s[/cyan]"
        )

        left_children = _split_recursive(left)
        right_children = _split_recursive(right)

        if left_children:
            left_children[-1]["end_reason"] = end_reason_on_split

        return left_children + right_children

    for seg in segments:
        children = _split_recursive(seg)
        for child in children:
            child["num"] = seg_num
            seg_num += 1
            result.append(child)

    return result
