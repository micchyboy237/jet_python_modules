# vad_speech_splitter.py

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechEndReason, SpeechSegment
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_HARD_LIMIT_MIN_TROUGH_OFFSET_S,
    DEFAULT_HARD_LIMIT_MIN_VALLEY_DURATION_S,
    DEFAULT_HARD_LIMIT_SEC,
    DEFAULT_HARD_LIMIT_SMOOTHING_WINDOW,
    DEFAULT_HARD_LIMIT_TROUGH_PROMINENCE,
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
)
from jet.audio.audio_waveform.vad.vad_utils import compute_hybrid_probs, make_segment
from jet.audio.helpers.config import (
    HOP_SIZE,
    HOP_STEP_S,
    SAMPLE_RATE,
)
from jet.audio.speech.vad_extractors import (
    get_best_valley_trough,
    split_best_valley_trough,
)
from jet.audio.speech.vad_types import ValleyTrough
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
    sample_rate: int,
    hop_sec: float,
    soft_limit_sec: float,
    return_seconds: bool,
    with_scores: bool,
    make_segment: Callable,
    smoothing_window: int = DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
    trough_prominence: float = DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
    min_valley_duration_s: float = DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
    min_trough_offset_s: float = DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
) -> List[SpeechSegment]:
    limit_configs = _get_limit_configs()
    result: List[SpeechSegment] = []
    seg_num = 1

    for seg in segments:
        if seg["type"] != "speech":
            seg["num"] = seg_num
            result.append(seg)
            seg_num += 1
            continue

        start_s: float = seg["start"] if return_seconds else seg["start"] / sample_rate
        end_s: float = seg["end"] if return_seconds else seg["end"] / sample_rate

        chosen_pieces: list[tuple[float, float]] = [(start_s, end_s)]
        chosen_config: LimitConfig | None = None
        chosen_trough: ValleyTrough | None = None

        for config in limit_configs:
            pieces, trough = _apply_single_limit_split(
                p_start=start_s,
                p_end=end_s,
                probs=probs,
                hop_sec=hop_sec,
                config=config,
            )
            if trough is not None:
                chosen_pieces = pieces
                chosen_config = config
                chosen_trough = trough
                break

        last_idx = len(chosen_pieces) - 1
        for i, (piece_start, piece_end) in enumerate(chosen_pieces):
            if chosen_config is None or len(chosen_pieces) == 1:
                piece_end_reason = seg.get("end_reason")
                piece_trough = None
            elif i < last_idx:
                piece_end_reason = chosen_config.end_reason
                piece_trough = chosen_trough
            else:
                piece_end_reason = "silent"
                piece_trough = None  # ← last piece has no valley trough; it ends for a different reason

            result.append(
                make_segment(
                    seg_num,
                    piece_start,
                    piece_end,
                    "speech",
                    end_reason=piece_end_reason,
                    is_ongoing=False,
                    last_non_speech_sec=None,
                    best_valley_trough=piece_trough,
                )
            )
            seg_num += 1

    return result


def _apply_single_limit_split(
    p_start: float,
    p_end: float,
    probs: List[float],
    hop_sec: float,
    config: LimitConfig,
) -> tuple[list[tuple[float, float]], "ValleyTrough | None"]:
    """
    Attempt to split the time range (p_start, p_end) at the best silence
    valley using a single LimitConfig's parameters.

    The range is split iteratively: each resulting piece that still exceeds
    config.max_limit_sec is queued for another split attempt under the same
    config.  Pieces that are already short enough, or for which no trough is
    found, are collected as-is.

    Uses get_best_valley_trough scoped to each sub-range's prob slice.
    The min_trough_offset_s from config is relied upon to ensure the trough
    sits meaningfully inside the range — no additional positional guard needed.

    Args:
        p_start: Range start in seconds.
        p_end: Range end in seconds.
        probs: Full-audio framewise VAD probabilities.
        hop_sec: Duration of one probability frame in seconds.
        config: Limit parameters (threshold, valley settings, end_reason).

    Returns:
        A tuple of:
          - split_pieces: list of (start_s, end_s) sub-ranges, always
            non-empty (at minimum [(p_start, p_end)] if nothing split).
          - best_trough: the ValleyTrough with the highest final_score that
            caused any split in this pass, or None if no split occurred.
    """
    pending: list[tuple[float, float]] = [(p_start, p_end)]
    split_pieces: list[tuple[float, float]] = []
    best_trough: "ValleyTrough | None" = None

    while pending:
        cur_start, cur_end = pending.pop(0)
        duration = cur_end - cur_start

        if duration <= config.max_limit_sec:
            split_pieces.append((cur_start, cur_end))
            continue

        # Slice probs to the current sub-range so detection is scoped correctly.
        frame_start = int(cur_start / hop_sec)
        frame_end = int(cur_end / hop_sec)
        seg_probs = probs[frame_start : frame_end + 1]

        if not seg_probs:
            split_pieces.append((cur_start, cur_end))
            continue

        # Use get_best_valley_trough scoped to seg_probs.
        # Switch to split_best_valley_trough if audio is ever threaded through.
        candidate: "ValleyTrough | None" = get_best_valley_trough(
            probs_or_audio=seg_probs,
            smoothing_window=0,
            trough_height=0.3,
            trough_prominence=0.0,
            min_valley_duration_s=0.1,
            min_trough_offset_s=2.0,
        )

        if candidate is None:
            # This config cannot split this sub-range.
            split_pieces.append((cur_start, cur_end))
            continue

        # candidate.time_s is LOCAL to seg_probs; shift back to global timeline.
        split_time_s: float = cur_start + candidate["time_s"]

        # Track the overall best trough across all splits in this pass.
        if best_trough is None or (
            candidate["valley"]["final_score"] > best_trough["valley"]["final_score"]
        ):
            best_trough = candidate

        # Queue both halves — each will be re-checked against the same limit.
        pending.insert(0, (split_time_s, cur_end))
        pending.insert(0, (cur_start, split_time_s))

    return split_pieces, best_trough


def split_segments(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    hop_sec: float = HOP_STEP_S,
    soft_limit_sec: float = DEFAULT_SOFT_LIMIT_SEC,
    smoothing_window: int = DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
    trough_prominence: float = DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
    min_valley_duration_s: float = DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
    min_trough_offset_s: float = DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
) -> tuple[List[SpeechSegment], List[np.ndarray]]:
    from jet.audio.speech.vad_extractors2 import load_vad_probs

    accumulated_segments: List[SpeechSegment] = []
    accumulated_chunks: List[np.ndarray] = []
    seg_num = 0

    def _inner_split_fn(
        audio_chunk: np.ndarray,
        frame_offset: int = 0,
        time_offset_s: float = 0.0,
    ) -> None:
        nonlocal seg_num

        split_result = split_best_valley_trough(
            probs_or_audio=audio_chunk,
            sample_rate=sample_rate,
            min_valley_duration_s=min_valley_duration_s,
            smoothing_window=smoothing_window,
            trough_prominence=trough_prominence,
            valley_threshold=None,
            min_trough_offset_s=2.0,
        )

        if split_result is None:
            # No further split possible — emit the whole chunk as one segment
            if len(audio_chunk) == 0:
                return

            # Build local probs from the audio chunk directly via load_vad_probs
            local_probs = load_vad_probs(audio_chunk)
            duration_s = len(audio_chunk) / sample_rate
            end_s = time_offset_s + duration_s

            seg_num += 1
            segment = make_segment(
                num=seg_num,
                start_sec=time_offset_s,
                end_sec=end_s,
                probs=local_probs,
                seg_type="speech",
                end_reason="hard_limit",
                is_ongoing=False,
                last_non_speech_sec=None,
                best_valley_trough=None,
                sample_rate=sample_rate,
                return_seconds=True,
                with_scores=True,
            )

            frame_start = frame_offset + segment["frame_start"]
            frame_end = frame_offset + segment["frame_end"]

            # Patch frame_start / frame_end to reflect global position
            segment["frame_start"] = frame_start
            segment["frame_end"] = frame_end
            accumulated_segments.append(segment)
            accumulated_chunks.append(audio_chunk)
            return

        left_probs, right_probs, best_trough, left_audio_np, right_audio_np = (
            split_result
        )

        # --- Left segment ---
        left_duration_s = len(left_audio_np) / sample_rate
        left_end_s = time_offset_s + left_duration_s
        left_split_frame = best_trough["frame"]  # local frame within this chunk

        seg_num += 1
        left_segment = make_segment(
            num=seg_num,
            start_sec=time_offset_s,
            end_sec=left_end_s,
            probs=left_probs,
            seg_type="speech",
            end_reason="valley",
            is_ongoing=False,
            last_non_speech_sec=None,
            best_valley_trough=best_trough,
            sample_rate=sample_rate,
            return_seconds=True,
            with_scores=True,
        )

        left_frame_start = frame_offset + left_segment["frame_start"]
        left_frame_end = frame_offset + left_segment["frame_end"]

        # Shift local frame indices to global
        left_segment["frame_start"] = left_frame_start
        left_segment["frame_end"] = left_frame_end
        accumulated_segments.append(left_segment)
        accumulated_chunks.append(left_audio_np)

        # --- Recurse on right chunk ---
        right_frame_offset = frame_offset + left_split_frame
        right_time_offset_s = time_offset_s + left_duration_s

        _inner_split_fn(
            audio_chunk=right_audio_np,
            frame_offset=right_frame_offset,
            time_offset_s=right_time_offset_s + hop_sec,
        )

    # Kick off recursion from the full audio
    _inner_split_fn(audio_chunk=audio, frame_offset=0, time_offset_s=0.0)
    return accumulated_segments, accumulated_chunks
