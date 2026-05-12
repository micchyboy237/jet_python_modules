from typing import List

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechEndReason, SpeechSegment
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_PROB_WEIGHT,
    DEFAULT_RMS_WEIGHT,
)
from jet.audio.audio_waveform.vad.vad_utils import compute_hybrid_probs
from jet.audio.helpers.config import (
    HOP_SIZE,
    HOP_STEP_S,
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
) -> int:
    """
    Given a speech-segment onset (in samples), look backward through the
    pre-speech audio and find how many additional samples to prepend.

    Strategy
    --------
    1. Build per-frame hybrid scores for up to *max_preroll_sec* before onset.
    2. Walk backward from the onset frame; extend the pre-roll for every
       consecutive frame whose hybrid score >= hybrid_threshold.
    3. Return the number of *samples* to prepend (>= 0).

    The hybrid score per 10 ms frame:
        score = prob_weight * smoothed_prob + rms_weight * rms_norm

    RMS is normalised using the 99th-percentile of the look-back window.
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

    hybrid = compute_hybrid_probs(
        probs=lookback_probs,
        audio_np=lookback_audio,
        prob_weight=prob_weight,
        rms_weight=rms_weight,
        frame_samples=HOP_SIZE,
    )

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
) -> int:
    """
    Given a speech-segment end (in samples), look *forward* through the
    post-speech audio and find how many additional samples to append.

    Strategy
    --------
    1. Build per-frame hybrid scores for up to *max_postroll_sec* after end.
    2. Walk forward from the end frame; extend the post-roll for every
       consecutive frame whose hybrid score >= hybrid_threshold.
    3. Return the number of *samples* to append (>= 0).

    The hybrid score per 10 ms frame:
        score = prob_weight * smoothed_prob + rms_weight * rms_norm

    RMS is normalised using the 99th-percentile of the look-forward window.
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

    hybrid = compute_hybrid_probs(
        probs=lookahead_probs,
        audio_np=lookahead_audio,
        prob_weight=prob_weight,
        rms_weight=rms_weight,
        frame_samples=HOP_SIZE,
    )

    # Walk forward from the frame immediately after the detected end
    keep_frames = 0
    for i in range(n_frames):
        if hybrid[i] >= hybrid_threshold:
            keep_frames = i + 1  # extend at least through this frame
        else:
            break  # stop at first sub-threshold frame

    return keep_frames * HOP_SIZE


def apply_limit_splits(
    segments: List[SpeechSegment],
    probs: List[float],
    audio_np: np.ndarray,  # Insert audio_np argument
    sample_rate: int,
    hop_sec: float,
    max_limit_sec: float,
    min_valley_duration_s: float,
    smoothing_window: int,
    trough_prominence: float,
    min_trough_offset_s: float,
    return_seconds: bool,
    with_scores: bool,
    end_reason_on_split: SpeechEndReason = "valley",
    hybrid_prob_weight: float = DEFAULT_PROB_WEIGHT,  # Add hybrid_prob_weight argument
    hybrid_rms_weight: float = DEFAULT_RMS_WEIGHT,  # Add hybrid_rms_weight argument
) -> List[SpeechSegment]:
    """
    Recursively split speech segments that exceed *max_limit_sec* by finding
    the best valley trough in the segment's probability slice and splitting there.

    Args:
        segments:              The current list of SpeechSegment dicts.
        probs:                 Full-audio framewise speech probabilities.
        audio_np:              Full audio signal as numpy array.
        sample_rate:           Audio sample rate (Hz).
        hop_sec:               Seconds per probability frame (typically 0.010).
        max_limit_sec:         Maximum preferred segment duration before splitting.
        min_valley_duration_s: Min silence width to qualify as a split candidate.
        smoothing_window:      Smoothing window passed to get_best_valley_trough.
        trough_prominence:     Min trough prominence for detection.
        min_trough_offset_s:   Trough must be >= this many seconds from segment start.
        return_seconds:        Whether segment start/end are in seconds or samples.
        end_reason_on_split:   Value set on the *left* child's end_reason when a valley split occurs.
        with_scores:           Whether segment_probs should be populated.
        hybrid_prob_weight:    Weight for probability when combining with RMS.
        hybrid_rms_weight:     Weight for RMS when combining with probability.

    Returns:
        New (possibly longer) list of SpeechSegment dicts with renumbered ``num``
        fields, long segments replaced by their split children.
    """
    from jet.audio.speech.vad_extractors import get_best_valley_trough

    result: List[SpeechSegment] = []
    seg_num = 1

    def _split_recursive(seg: SpeechSegment) -> List[SpeechSegment]:
        """Return one or more segments produced from *seg*, splitting if needed."""
        duration = seg["duration"]
        if duration <= max_limit_sec:
            return [seg]

        # Extract the probability slice for this segment
        frame_start: int = seg["frame_start"]
        frame_end: int = seg["frame_end"]

        # --- Start Hybrid Score Calculation (from file_context_0) ---
        seg_audio = audio_np[int(frame_start * 160) : int((frame_end + 1) * 160)]
        segment_probs = np.array(probs[frame_start : frame_end + 1], dtype=np.float32)
        seg_probs = compute_hybrid_probs(
            probs=segment_probs,
            audio_np=seg_audio,
            prob_weight=hybrid_prob_weight,
            rms_weight=hybrid_rms_weight,
        ).tolist()
        # --- End Hybrid Score Calculation ---

        if not seg_probs:
            return [seg]

        # Find the best valley trough inside this segment
        best_trough = get_best_valley_trough(
            probs=seg_probs,
            smoothing_window=smoothing_window,
            trough_prominence=trough_prominence,
            min_valley_duration_s=min_valley_duration_s,
            min_trough_offset_s=min_trough_offset_s,
        )

        if best_trough is None:
            # No suitable silence found — return the long segment as-is
            console.print(
                f"[yellow]Soft limit: segment {seg['num']} is {duration:.1f}s "
                f"(>{max_limit_sec}s) but no valley trough found — keeping intact.[/yellow]"
            )
            return [seg]

        # Trough frame is local to seg_probs; convert to global frame index
        local_trough_frame: int = best_trough["frame"]
        global_trough_frame: int = frame_start + local_trough_frame
        split_time_s: float = global_trough_frame * hop_sec

        # Build left and right child segments
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
                num=0,  # renumbered after recursion
                start=start_val,
                end=end_val,
                prob=avg_prob,
                duration=duration_s,
                frames_length=len(child_probs_slice),
                frame_start=child_frame_start,
                frame_end=child_frame_end,
                type=seg["type"],
                segment_probs=child_probs_slice if with_scores else [],
                end_reason=None,  # filled in by caller after recursion
            )

        left = _make_child(frame_start, global_trough_frame)
        right = _make_child(global_trough_frame, frame_end)

        console.print(
            f"[cyan]Soft limit: split segment {seg['num']} "
            f"({duration:.1f}s) at {split_time_s:.2f}s "
            f"→ {left['duration']:.1f}s + {right['duration']:.1f}s[/cyan]"
        )

        # Recurse on each half independently
        # Mark the left child: it ended at a valley boundary.
        # The right child inherits the original segment's end_reason.
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
