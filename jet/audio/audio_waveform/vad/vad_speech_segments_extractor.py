import json
from pathlib import Path
from typing import List, Literal, Union

import matplotlib
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.audio_waveform.vad.vad_firered_hybrid import FireRedVAD
from jet.audio.audio_waveform.vad.vad_utils import compute_hybrid_probs, save_segments
from jet.audio.helpers.config import (
    HOP_SIZE,
    HOP_STEP_S,
    SAMPLE_RATE,
)
from jet.audio.speech.vad_extractors import get_best_valley_trough
from jet.audio.utils.loader import load_audio

matplotlib.use("Agg")
import numpy as np
import torch
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_INCLUDE_NON_SPEECH,
    DEFAULT_MAX_BUFFER_SEC,
    DEFAULT_MAX_SPEECH_SEC,
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_MIN_SPEECH_SEC,
    DEFAULT_POSTROLL_HYBRID_THRESHOLD,
    DEFAULT_POSTROLL_MAX_SEC,
    DEFAULT_PREROLL_HYBRID_THRESHOLD,
    DEFAULT_PREROLL_MAX_SEC,
    DEFAULT_PROB_WEIGHT,
    DEFAULT_RETURN_SECONDS,
    DEFAULT_RMS_WEIGHT,
    DEFAULT_SAMPLING_RATE,
    DEFAULT_SMOOTH_WINDOW_SIZE,
    DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
    DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
    DEFAULT_SOFT_LIMIT_SEC,
    DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
    DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
    DEFAULT_THRESHOLD,
    DEFAULT_WITH_SCORES,
)
from rich.console import Console

console = Console()

SAVE_DIR = str(
    Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD").expanduser().resolve()
)


# ---------------------------------------------------------------------------
# Pre-roll computation helper
# ---------------------------------------------------------------------------


def _compute_preroll(
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


def _compute_postroll(
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


def _apply_soft_limit_splits(
    segments: List[SpeechSegment],
    probs: List[float],
    audio_np: np.ndarray,  # Insert audio_np argument
    sample_rate: int,
    hop_sec: float,
    soft_limit_sec: float,
    min_valley_duration_s: float,
    smoothing_window: int,
    trough_prominence: float,
    min_trough_offset_s: float,
    return_seconds: bool,
    with_scores: bool,
    hybrid_prob_weight: float = DEFAULT_PROB_WEIGHT,  # Add hybrid_prob_weight argument
    hybrid_rms_weight: float = DEFAULT_RMS_WEIGHT,  # Add hybrid_rms_weight argument
) -> List[SpeechSegment]:
    """
    Recursively split speech segments that exceed *soft_limit_sec* by finding
    the best valley trough in the segment's probability slice and splitting there.

    Args:
        segments:              The current list of SpeechSegment dicts.
        probs:                 Full-audio framewise speech probabilities.
        audio_np:              Full audio signal as numpy array.
        sample_rate:           Audio sample rate (Hz).
        hop_sec:               Seconds per probability frame (typically 0.010).
        soft_limit_sec:        Maximum preferred segment duration before splitting.
        min_valley_duration_s: Min silence width to qualify as a split candidate.
        smoothing_window:      Smoothing window passed to get_best_valley_trough.
        trough_prominence:     Min trough prominence for detection.
        min_trough_offset_s:   Trough must be >= this many seconds from segment start.
        return_seconds:        Whether segment start/end are in seconds or samples.
        with_scores:           Whether segment_probs should be populated.
        hybrid_prob_weight:    Weight for probability when combining with RMS.
        hybrid_rms_weight:     Weight for RMS when combining with probability.

    Returns:
        New (possibly longer) list of SpeechSegment dicts with renumbered ``num``
        fields, long segments replaced by their split children.
    """
    result: List[SpeechSegment] = []
    seg_num = 1

    def _split_recursive(seg: SpeechSegment) -> List[SpeechSegment]:
        """Return one or more segments produced from *seg*, splitting if needed."""
        duration = seg["duration"]
        if duration <= soft_limit_sec:
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
                f"(>{soft_limit_sec}s) but no valley trough found — keeping intact.[/yellow]"
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
            )

        left = _make_child(frame_start, global_trough_frame)
        right = _make_child(global_trough_frame, frame_end)

        console.print(
            f"[cyan]Soft limit: split segment {seg['num']} "
            f"({duration:.1f}s) at {split_time_s:.2f}s "
            f"→ {left['duration']:.1f}s + {right['duration']:.1f}s[/cyan]"
        )

        # Recurse on each half independently
        return _split_recursive(left) + _split_recursive(right)

    for seg in segments:
        children = _split_recursive(seg)
        for child in children:
            child["num"] = seg_num
            seg_num += 1
            result.append(child)

    return result


def extract_speech_timestamps(
    audio: Union[str, Path, np.ndarray, torch.Tensor, list[np.ndarray]],
    threshold: float = DEFAULT_THRESHOLD,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_speech_duration_sec: float | None = None,
    return_seconds: bool = DEFAULT_RETURN_SECONDS,
    with_scores: bool = DEFAULT_WITH_SCORES,
    include_non_speech: bool = DEFAULT_INCLUDE_NON_SPEECH,
    smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
    max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
    preroll_max_sec: float = DEFAULT_PREROLL_MAX_SEC,
    preroll_hybrid_threshold: float = DEFAULT_PREROLL_HYBRID_THRESHOLD,
    preroll_prob_weight: float = DEFAULT_PROB_WEIGHT,
    preroll_rms_weight: float = DEFAULT_RMS_WEIGHT,
    postroll_max_sec: float = DEFAULT_POSTROLL_MAX_SEC,
    postroll_hybrid_threshold: float = DEFAULT_POSTROLL_HYBRID_THRESHOLD,
    postroll_prob_weight: float = DEFAULT_PROB_WEIGHT,
    postroll_rms_weight: float = DEFAULT_RMS_WEIGHT,
    soft_limit_sec: float = DEFAULT_SOFT_LIMIT_SEC,
    soft_limit_min_valley_duration_s: float = DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
    soft_limit_smoothing_window: int = DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
    soft_limit_trough_prominence: float = DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
    soft_limit_min_trough_offset_s: float = DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
    **kwargs,
) -> Union[List[SpeechSegment], tuple[List[SpeechSegment], List[float]]]:
    """
    Extract speech timestamps using FireRedVAD with symmetric hybrid
    pre-roll (head) and post-roll (tail) boundary extension.

    When a speech segment exceeds *soft_limit_sec*, valley trough detection
    is used to find the best natural silence and split the segment there.
    Splitting is recursive: each half is re-checked until no segment exceeds
    the soft limit or no trough can be found.

    Both boundaries are extended by a variable amount computed from a
    weighted combination of smoothed speech probability and normalised RMS
    energy (equal 0.5/0.5 weights by default).

    When include_non_speech=True, returns both speech and non-speech segments.
    """
    if max_speech_duration_sec is None:
        max_speech_duration_sec = DEFAULT_MAX_SPEECH_SEC

    audio_np, sr = load_audio(audio, sr=SAMPLE_RATE, mono=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"FireRedVAD requires SAMPLE_RATE Hz, got {sr}")

    vad = FireRedVAD(
        model_dir=SAVE_DIR,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
        smooth_window_size=smooth_window_size,
        max_buffer_sec=max_buffer_sec,
    )

    with console.status("[bold blue]Running FireRedVAD inference...[/bold blue]"):
        frame_results, result = vad.detect_full(audio_np)

    timestamps = result["timestamps"]
    probs = [r.smoothed_prob for r in frame_results]

    # ------------------------------------------------------------------
    # Apply hybrid pre-roll (head) and post-roll (tail) to each segment
    # ------------------------------------------------------------------
    extended_timestamps: list[tuple[float, float]] = []
    total_samples = len(audio_np)
    for start_sec, end_sec in timestamps:
        onset_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)

        preroll_samples = _compute_preroll(
            onset_sample=onset_sample,
            audio_np=audio_np,
            probs=probs,
            sample_rate=sr,
            max_preroll_sec=preroll_max_sec,
            hybrid_threshold=preroll_hybrid_threshold,
            prob_weight=preroll_prob_weight,
            rms_weight=preroll_rms_weight,
        )
        postroll_samples = _compute_postroll(
            end_sample=end_sample,
            audio_np=audio_np,
            probs=probs,
            sample_rate=sr,
            max_postroll_sec=postroll_max_sec,
            hybrid_threshold=postroll_hybrid_threshold,
            prob_weight=postroll_prob_weight,
            rms_weight=postroll_rms_weight,
        )

        new_start_sec = max(0.0, (onset_sample - preroll_samples) / sr)
        new_end_sec = min(total_samples / sr, (end_sample + postroll_samples) / sr)
        extended_timestamps.append((new_start_sec, new_end_sec))

    # Merge overlapping segments that may arise after pre-roll extension
    merged: list[tuple[float, float]] = []
    for seg in extended_timestamps:
        if merged and seg[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(list(seg))
    timestamps = [tuple(s) for s in merged]

    # ------------------------------------------------------------------
    # Build SpeechSegment objects
    # ------------------------------------------------------------------
    def make_segment(
        num: int,
        start_sec: float,
        end_sec: float,
        seg_type: Literal["speech", "non-speech"],
    ) -> SpeechSegment:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        frame_start = int(start_sec / HOP_STEP_S)
        frame_end = int(end_sec / HOP_STEP_S)
        segment_probs_slice = probs[frame_start : frame_end + 1]
        avg_prob = float(np.mean(segment_probs_slice)) if segment_probs_slice else 0.0
        duration_sec = end_sec - start_sec
        start_val = start_sec if return_seconds else start_sample
        end_val = end_sec if return_seconds else end_sample
        return SpeechSegment(
            num=num,
            start=start_val,
            end=end_val,
            prob=avg_prob,
            duration=duration_sec,
            frames_length=len(segment_probs_slice),
            frame_start=frame_start,
            frame_end=frame_end,
            type=seg_type,
            segment_probs=segment_probs_slice if with_scores else [],
        )

    enhanced: List[SpeechSegment] = []
    current_time = 0.0
    seg_num = 1

    if include_non_speech and timestamps and timestamps[0][0] > 0.001:
        enhanced.append(make_segment(seg_num, 0.0, timestamps[0][0], "non-speech"))
        seg_num += 1
        current_time = timestamps[0][0]

    for start_sec, end_sec in timestamps:
        if include_non_speech and start_sec > current_time + 0.01:
            enhanced.append(
                make_segment(seg_num, current_time, start_sec, "non-speech")
            )
            seg_num += 1
        enhanced.append(make_segment(seg_num, start_sec, end_sec, "speech"))
        seg_num += 1
        current_time = end_sec

    total_duration = result["dur"]
    if include_non_speech and current_time < total_duration - 0.01:
        enhanced.append(
            make_segment(seg_num, current_time, total_duration, "non-speech")
        )

    # ------------------------------------------------------------------
    # Soft-limit: split long segments at valley troughs
    # ------------------------------------------------------------------
    if soft_limit_sec > 0:
        enhanced = _apply_soft_limit_splits(
            segments=enhanced,
            probs=probs,
            audio_np=audio_np,
            sample_rate=sr,
            hop_sec=HOP_STEP_S,
            soft_limit_sec=soft_limit_sec,
            min_valley_duration_s=soft_limit_min_valley_duration_s,
            smoothing_window=soft_limit_smoothing_window,
            trough_prominence=soft_limit_trough_prominence,
            min_trough_offset_s=soft_limit_min_trough_offset_s,
            return_seconds=return_seconds,
            with_scores=with_scores,
        )

    if with_scores:
        return enhanced, probs
    return enhanced


# ---------------------------------------------------------------------------
# extract_speech_audio
# ---------------------------------------------------------------------------


def extract_speech_audio(
    audio: Union[str, Path, np.ndarray, torch.Tensor, list[np.ndarray]],
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    threshold: float = DEFAULT_THRESHOLD,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_speech_duration_sec: float | None = None,
    smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
    max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
    preroll_max_sec: float = DEFAULT_PREROLL_MAX_SEC,
    preroll_hybrid_threshold: float = DEFAULT_PREROLL_HYBRID_THRESHOLD,
    preroll_prob_weight: float = DEFAULT_PROB_WEIGHT,
    preroll_rms_weight: float = DEFAULT_RMS_WEIGHT,
    postroll_max_sec: float = DEFAULT_POSTROLL_MAX_SEC,
    postroll_hybrid_threshold: float = DEFAULT_POSTROLL_HYBRID_THRESHOLD,
    postroll_prob_weight: float = DEFAULT_PROB_WEIGHT,
    postroll_rms_weight: float = DEFAULT_RMS_WEIGHT,
) -> List[np.ndarray]:
    """
    Extract contiguous speech segments from the input audio using FireRedVAD.

    Both the head (onset) and tail (end) of each segment are extended via a
    hybrid (prob + RMS) pre-roll / post-roll before slicing the audio.

    Returns a flat list of numpy arrays where each array represents one
    complete speech segment in float32 format, normalised to [-1.0, 1.0].
    """
    if sampling_rate != SAMPLE_RATE:
        raise ValueError(f"FireRedVAD requires SAMPLE_RATE Hz, got {sampling_rate}")

    speech_segments = extract_speech_timestamps(
        audio=audio,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
        return_seconds=True,
        include_non_speech=False,
        smooth_window_size=smooth_window_size,
        max_buffer_sec=max_buffer_sec,
        preroll_max_sec=preroll_max_sec,
        preroll_hybrid_threshold=preroll_hybrid_threshold,
        preroll_prob_weight=preroll_prob_weight,
        preroll_rms_weight=preroll_rms_weight,
        postroll_max_sec=postroll_max_sec,
        postroll_hybrid_threshold=postroll_hybrid_threshold,
        postroll_prob_weight=postroll_prob_weight,
        postroll_rms_weight=postroll_rms_weight,
    )

    audio_np, sr = load_audio(audio=audio, sr=sampling_rate, mono=True)
    if sr != sampling_rate:
        raise ValueError(
            f"Loaded sample rate {sr} does not match requested {sampling_rate}"
        )

    speech_audio_chunks: List[np.ndarray] = []
    for segment in speech_segments:
        start_sec: float = segment["start"]
        end_sec: float = segment["end"]
        start_sample = int(round(start_sec * sr))
        end_sample = int(round(end_sec * sr))
        segment_audio = audio_np[start_sample:end_sample]
        if len(segment_audio) == 0:
            continue
        speech_audio_chunks.append(segment_audio.astype(np.float32, copy=False))

    return speech_audio_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import shutil

    DEFAULT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_1_speaker.wav"
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    parser = argparse.ArgumentParser(
        description="Extract speech segments with FireRedVAD + hybrid pre-roll"
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=DEFAULT_AUDIO,
        help="input audio file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(OUTPUT_DIR),
        type=str,
        help=f"output directory (default: '{OUTPUT_DIR}')",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"speech threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "-ms",
        "--min-silence",
        type=float,
        default=DEFAULT_MIN_SILENCE_SEC,
        help=f"minimum silence duration in seconds (default: {DEFAULT_MIN_SILENCE_SEC})",
    )
    parser.add_argument(
        "-mp",
        "--min-speech",
        type=float,
        default=DEFAULT_MIN_SPEECH_SEC,
        help=f"minimum speech duration in seconds (default: {DEFAULT_MIN_SPEECH_SEC})",
    )
    parser.add_argument(
        "-mx",
        "--max-speech",
        type=float,
        default=8.0,
        help="maximum speech duration in seconds",
    )
    parser.add_argument(
        "-sw",
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW_SIZE,
        help=f"smoothing window size (default: {DEFAULT_SMOOTH_WINDOW_SIZE})",
    )
    parser.add_argument(
        "-mb",
        "--max-buffer-sec",
        type=float,
        default=DEFAULT_MAX_BUFFER_SEC,
        help=f"stream buffer duration in seconds (default: {DEFAULT_MAX_BUFFER_SEC})",
    )
    # Pre-roll args
    parser.add_argument(
        "--preroll-max-sec",
        type=float,
        default=DEFAULT_PREROLL_MAX_SEC,
        help=f"max pre-roll look-back in seconds (default: {DEFAULT_PREROLL_MAX_SEC})",
    )
    parser.add_argument(
        "--preroll-threshold",
        type=float,
        default=DEFAULT_PREROLL_HYBRID_THRESHOLD,
        help=f"hybrid score threshold for pre-roll extension (default: {DEFAULT_PREROLL_HYBRID_THRESHOLD})",
    )
    parser.add_argument(
        "--preroll-prob-weight",
        type=float,
        default=DEFAULT_PROB_WEIGHT,
        help=f"weight for speech prob in hybrid score (default: {DEFAULT_PROB_WEIGHT})",
    )
    parser.add_argument(
        "--preroll-rms-weight",
        type=float,
        default=DEFAULT_RMS_WEIGHT,
        help=f"weight for RMS energy in hybrid score (default: {DEFAULT_RMS_WEIGHT})",
    )
    # Post-roll args
    parser.add_argument(
        "--postroll-max-sec",
        type=float,
        default=DEFAULT_POSTROLL_MAX_SEC,
        help=f"max post-roll look-forward in seconds (default: {DEFAULT_POSTROLL_MAX_SEC})",
    )
    parser.add_argument(
        "--postroll-threshold",
        type=float,
        default=DEFAULT_POSTROLL_HYBRID_THRESHOLD,
        help=f"hybrid score threshold for post-roll extension (default: {DEFAULT_POSTROLL_HYBRID_THRESHOLD})",
    )
    parser.add_argument(
        "--postroll-prob-weight",
        type=float,
        default=DEFAULT_PROB_WEIGHT,
        help=f"weight for speech prob in hybrid score (default: {DEFAULT_PROB_WEIGHT})",
    )
    parser.add_argument(
        "--postroll-rms-weight",
        type=float,
        default=DEFAULT_RMS_WEIGHT,
        help=f"weight for RMS energy in hybrid score (default: {DEFAULT_RMS_WEIGHT})",
    )
    parser.add_argument(
        "--soft-limit",
        type=float,
        default=DEFAULT_SOFT_LIMIT_SEC,
        help=f"Soft max segment duration before valley-trough splitting (default: {DEFAULT_SOFT_LIMIT_SEC}s; 0 = disabled)",
    )
    parser.add_argument(
        "--soft-limit-min-valley",
        type=float,
        default=DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
        help="Min silence duration for a split-candidate valley (default: %(default)ss)",
    )
    parser.add_argument(
        "--soft-limit-smoothing",
        type=int,
        default=DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
        help="Smoothing window for soft-limit trough detection (default: %(default)s frames)",
    )
    parser.add_argument(
        "--soft-limit-prominence",
        type=float,
        default=DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
        help="Min trough prominence for soft-limit splitting (default: %(default)s)",
    )
    parser.add_argument(
        "--soft-limit-offset",
        type=float,
        default=DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
        help="Trough must be >= this many seconds into the segment (default: %(default)ss)",
    )

    args = parser.parse_args()
    audio_path = args.audio_path
    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)

    console.rule("Audio Segmenter – FireRedVAD2 + Hybrid Pre/Post-Roll", style="blue")
    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_path).name}\n")
    console.print(
        f"[dim]Pre-roll:  max={args.preroll_max_sec}s  "
        f"threshold={args.preroll_threshold}  "
        f"prob_w={args.preroll_prob_weight}  "
        f"rms_w={args.preroll_rms_weight}[/dim]"
    )
    console.print(
        f"[dim]Post-roll: max={args.postroll_max_sec}s  "
        f"threshold={args.postroll_threshold}  "
        f"prob_w={args.postroll_prob_weight}  "
        f"rms_w={args.postroll_rms_weight}[/dim]\n"
    )

    # ── Step 1: detect segments (with per-frame probabilities) ────────────
    segments, speech_probs = extract_speech_timestamps(
        audio_path,
        threshold=args.threshold,
        min_silence_duration_sec=args.min_silence,
        min_speech_duration_sec=args.min_speech,
        max_speech_duration_sec=args.max_speech,
        return_seconds=True,
        with_scores=True,
        include_non_speech=False,
        smooth_window_size=args.smooth_window,
        max_buffer_sec=args.max_buffer_sec,
        preroll_max_sec=args.preroll_max_sec,
        preroll_hybrid_threshold=args.preroll_threshold,
        preroll_prob_weight=args.preroll_prob_weight,
        preroll_rms_weight=args.preroll_rms_weight,
        postroll_max_sec=args.postroll_max_sec,
        postroll_hybrid_threshold=args.postroll_threshold,
        postroll_prob_weight=args.postroll_prob_weight,
        postroll_rms_weight=args.postroll_rms_weight,
    )

    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments)}\n")
    for seg in segments:
        seg_type = seg["type"]
        type_color = "bold green" if seg_type == "speech" else "bold red"
        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white]"
            f" - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"dur=[bold magenta]{seg['duration']:.2f}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan] "
            f"type=[{type_color}]{seg_type}[/{type_color}]"
        )

    if not any(s["type"] == "speech" for s in segments):
        console.print("[red]No speech segments found.[/red]")
        raise SystemExit(0)

    # ── Step 2: extract raw audio for each speech segment ─────────────────
    audio_chunks = extract_speech_audio(
        audio_path,
        sampling_rate=DEFAULT_SAMPLING_RATE,
        threshold=args.threshold,
        min_silence_duration_sec=args.min_silence,
        min_speech_duration_sec=args.min_speech,
        max_speech_duration_sec=args.max_speech,
        smooth_window_size=args.smooth_window,
        max_buffer_sec=args.max_buffer_sec,
        preroll_max_sec=args.preroll_max_sec,
        preroll_hybrid_threshold=args.preroll_threshold,
        preroll_prob_weight=args.preroll_prob_weight,
        preroll_rms_weight=args.preroll_rms_weight,
        postroll_max_sec=args.postroll_max_sec,
        postroll_hybrid_threshold=args.postroll_threshold,
        postroll_prob_weight=args.postroll_prob_weight,
        postroll_rms_weight=args.postroll_rms_weight,
    )

    # ── Step 3: save everything to disk ───────────────────────────────────
    saved_metas = save_segments(segments, audio_chunks, output_dir)

    # ── Step 4: write summary JSON files ──────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "all_speech_segments.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        slim = [
            {k: v for k, v in m.items() if k != "segment_probs"} for m in saved_metas
        ]
        json.dump(slim, fh, ensure_ascii=False, indent=2)
    console.print(
        f"[bold green]✓ Summary saved to:[/bold green] "
        f"[link=file://{summary_path.resolve()}]{summary_path}[/link]"
    )

    all_probs_path = output_dir / "speech_probs.json"
    with open(all_probs_path, "w", encoding="utf-8") as fh:
        json.dump(
            speech_probs if isinstance(speech_probs, list) else [],
            fh,
            indent=2,
        )
    console.print(
        f"[bold green]✓ Full probs saved to:[/bold green] "
        f"[link=file://{all_probs_path.resolve()}]{all_probs_path}[/link]"
    )

    console.rule("Done", style="green")
