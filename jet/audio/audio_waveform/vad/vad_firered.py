import json
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from fireredvad.core.constants import SAMPLE_RATE
from fireredvad.core.stream_vad_postprocessor import StreamVadFrameResult
from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_USE_HYBRID,
)
from jet.audio.normalization.dtype_conversion import convert_audio_dtype
from jet.audio.utils.info import display_audio_info
from jet.audio.utils.loader import load_audio
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()

SAVE_DIR = str(
    Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD").expanduser().resolve()
)

DEFAULT_THRESHOLD = 0.3
DEFAULT_NEG_THRESHOLD = 0.1
DEFAULT_MIN_SILENCE_SEC = 0.250
DEFAULT_MIN_SPEECH_SEC = 0.250
DEFAULT_MAX_SPEECH_SEC = None
DEFAULT_SAMPLING_RATE = 16000
DEFAULT_RETURN_SECONDS = False
DEFAULT_WITH_SCORES = False
DEFAULT_INCLUDE_NON_SPEECH = False

DEFAULT_SMOOTH_WINDOW_SIZE = 5
DEFAULT_PAD_START_FRAME = 5
DEFAULT_MAX_BUFFER_SEC = 1.2

DEFAULT_PROB_WEIGHT = 0.5
DEFAULT_RMS_WEIGHT = 0.5


# Single global cached instance to avoid repeated model loading
_global_vad_cache: Optional["FireRedVAD"] = None
_global_vad_cache_config: Optional[dict] = None  # Track the config used


def get_global_vad(
    threshold: float = DEFAULT_THRESHOLD,
    neg_threshold: float = DEFAULT_NEG_THRESHOLD,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_speech_duration_sec: float | None = DEFAULT_MAX_SPEECH_SEC,  # Now accepts None
    smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
    pad_start_frame: int = DEFAULT_PAD_START_FRAME,
    max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
    use_hybrid: bool = DEFAULT_USE_HYBRID,
    **model_kwargs,
) -> "FireRedVAD":
    """Get or create the global cached VAD instance."""
    global _global_vad_cache, _global_vad_cache_config

    current_config = {
        "threshold": threshold,
        "neg_threshold": neg_threshold,
        "min_silence_duration_sec": min_silence_duration_sec,
        "min_speech_duration_sec": min_speech_duration_sec,
        "max_speech_duration_sec": max_speech_duration_sec,  # Can be None
        "smooth_window_size": smooth_window_size,
        "pad_start_frame": pad_start_frame,
        "max_buffer_sec": max_buffer_sec,
        "use_hybrid": use_hybrid,
    }
    current_config.update(model_kwargs)

    if _global_vad_cache is None:
        with console.status(
            "[bold blue]Loading FireRedVAD model (global cache)...[/bold blue]"
        ):
            _global_vad_cache = FireRedVAD(
                model_dir=SAVE_DIR,
                threshold=threshold,
                neg_threshold=neg_threshold,
                min_silence_duration_sec=min_silence_duration_sec,
                min_speech_duration_sec=min_speech_duration_sec,
                max_speech_duration_sec=max_speech_duration_sec,  # Can be None
                smooth_window_size=smooth_window_size,
                pad_start_frame=pad_start_frame,
                max_buffer_sec=max_buffer_sec,
                use_hybrid=use_hybrid,
                **model_kwargs,
            )
            _global_vad_cache_config = current_config
    else:
        if _global_vad_cache_config != current_config:
            changed_params = {
                k: (v, current_config[k])
                for k, v in _global_vad_cache_config.items()
                if k in current_config and v != current_config[k]
            }
            console.print(
                f"[yellow]Warning: VAD parameters differ from cached model. "
                f"Reusing existing model. Changed params: {changed_params}. "
                f"Call clear_global_vad_cache() first if you need the new parameters.[/yellow]"
            )

    return _global_vad_cache


def clear_global_vad_cache() -> None:
    """Clear the global VAD cache, forcing a fresh model load on next use."""
    global _global_vad_cache, _global_vad_cache_config
    if _global_vad_cache is not None:
        # Clean up GPU memory if needed
        del _global_vad_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _global_vad_cache = None
    _global_vad_cache_config = None
    console.print("[dim]VAD global cache cleared.[/dim]")


class FireRedVAD:
    """Wrapper for FireRedVAD with simple streaming-like API."""

    def __init__(
        self,
        model_dir: str = SAVE_DIR,
        device: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        neg_threshold: float = DEFAULT_NEG_THRESHOLD,
        min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
        min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
        max_speech_duration_sec: float
        | None = DEFAULT_MAX_SPEECH_SEC,  # Now accepts None
        smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
        pad_start_frame: int = DEFAULT_PAD_START_FRAME,
        max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
        use_hybrid: bool = DEFAULT_USE_HYBRID,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.neg_threshold = neg_threshold

        console.print(f"[cyan]Loading FireRedVAD (streaming) on {self.device}…[/cyan]")
        console.print(
            f"[cyan]Neg threshold: {self.neg_threshold}, Max speech duration: {max_speech_duration_sec}[/cyan]"
        )

        frames_per_sec = 100

        # Convert max_speech_duration_sec to frames, or use a very large value if None
        if max_speech_duration_sec is None:
            max_speech_frame = 30000  # Large enough to effectively be unlimited
        else:
            max_speech_frame = int(max_speech_duration_sec * frames_per_sec)

        config = FireRedStreamVadConfig(
            use_gpu=(device == "cuda"),
            speech_threshold=threshold,
            smooth_window_size=smooth_window_size,
            pad_start_frame=pad_start_frame,
            min_speech_frame=int(min_speech_duration_sec * frames_per_sec),
            max_speech_frame=max_speech_frame,
            min_silence_frame=int(min_silence_duration_sec * frames_per_sec),
            chunk_max_frame=30000,
        )
        self.vad = FireRedStreamVad.from_pretrained(model_dir, config=config)
        self.vad.vad_model.to(self.device)
        console.print("[green]done.[/green]")

        self.sample_rate = SAMPLE_RATE
        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.last_prob: float = 0.0
        self.max_buffer_samples = int(max_buffer_sec * self.sample_rate)

        # Track segments for neg threshold logic
        self._current_segment_probs: List[float] = []
        self._in_segment: bool = False

        self._use_hybrid = use_hybrid

    def reset(self) -> None:
        """Reset internal VAD state and clear audio buffer."""
        self.vad.reset()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_prob = 0.0
        self._current_segment_probs = []  # NEW
        self._in_segment = False  # NEW

    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Simple dynamic range compression / gain normalization."""
        if len(chunk) == 0:
            return chunk.astype(np.float32)
        chunk = chunk.astype(np.float32)
        chunk_max = np.max(np.abs(chunk)) + 1e-10
        target_peak = 0.30
        if chunk_max < 0.20:
            gain = min(target_peak / chunk_max, 8.0)
            chunk = chunk * gain
        elif chunk_max > 0.60:
            gain = 0.60 / chunk_max
            chunk = chunk * gain
        return chunk

    @torch.inference_mode()
    def get_speech_prob(self, chunk: np.ndarray) -> float:
        """
        Process incoming audio chunk (any length) and return
        the **latest smoothed speech probability**.
        """
        if len(chunk) == 0:
            return self.last_prob
        chunk = self._normalize_chunk(chunk)
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        if len(self.audio_buffer) < 4800:
            return self.last_prob
        to_process = self.audio_buffer[-9600:]
        results = self.vad.detect_chunk(to_process)
        self.audio_buffer = self.audio_buffer[-512:]
        if not results:
            return self.last_prob
        last = results[-1]
        prob = last.smoothed_prob
        self.last_prob = prob
        return prob

    def get_latest_result(self) -> Optional[dict]:
        """
        Optional: return more detailed info about the last processed frame
        (useful for debugging or when you need is_speech_start / is_speech_end).
        """
        return None

    def detect_full(self, audio):
        self.reset()
        if self._use_hybrid:
            frame_results, result = self.detect_full_hybrid(audio)
        else:
            frame_results, result = self.vad.detect_full(audio)
        if self.neg_threshold > 0:
            frame_results = self._apply_neg_threshold_to_results(frame_results)
            result["timestamps"] = FireRedStreamVad.results_to_timestamps(frame_results)
        return frame_results, result

    def detect_full_hybrid(
        self,
        audio: Union[str, np.ndarray],
        prob_weight: float = DEFAULT_PROB_WEIGHT,
        rms_weight: float = DEFAULT_RMS_WEIGHT,
    ) -> Tuple[List[StreamVadFrameResult], dict]:
        """
        detect_full variant that blends model probs with RMS energy.

        Strategy:
        1. Run FireRedStreamVad.detect_full normally to get raw model probs.
        2. Load audio samples for RMS computation.
        3. Compute hybrid probs via compute_hybrid_probs.
        4. Reset postprocessor and re-feed hybrid probs frame-by-frame so the
            state machine operates on blended values.
        5. Re-derive timestamps from the hybrid results.

        This avoids duplicating the model-forward / chunking logic from
        FireRedStreamVad.detect_full.
        """
        import logging

        import soundfile as sf
        from jet.audio.audio_waveform.vad.vad_utils import compute_hybrid_probs

        logger = logging.getLogger(__name__)

        # --- Step 1: base inference (always non-hybrid at the FireRedStreamVad level) ---
        frame_results, result = self.vad.detect_full(audio)

        raw_probs = [r.raw_prob for r in frame_results]
        logger.debug(
            "detect_full_hybrid: got %d raw model probs from base VAD",
            len(raw_probs),
        )

        # --- Step 2: load audio for RMS ---
        if isinstance(audio, str):
            audio_np, _ = sf.read(audio)
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
        else:
            audio_np = audio

        # --- Step 3: compute hybrid probs ---
        hybrid_probs = compute_hybrid_probs(
            probs=raw_probs,
            audio_np=audio_np,
            prob_weight=prob_weight,
            rms_weight=rms_weight,
        )
        logger.debug(
            "detect_full_hybrid: computed %d hybrid probs (prob_w=%.2f, rms_w=%.2f)",
            len(hybrid_probs),
            prob_weight,
            rms_weight,
        )

        # --- Step 4: re-run postprocessor with hybrid values ---
        self.vad.postprocessor.reset()
        hybrid_frame_results: List[StreamVadFrameResult] = []
        for hybrid_prob in hybrid_probs:
            hfr = self.vad.postprocessor.process_one_frame(float(hybrid_prob))
            hybrid_frame_results.append(hfr)

        # --- Step 5: re-derive timestamps ---
        timestamps = FireRedStreamVad.results_to_timestamps(hybrid_frame_results)
        hybrid_result = {
            "dur": result["dur"],
            "timestamps": timestamps,
        }
        if isinstance(audio, str):
            hybrid_result["wav_path"] = audio

        logger.info(
            "detect_full_hybrid: blended %d frames → %d speech segments",
            len(hybrid_frame_results),
            len(timestamps),
        )
        return hybrid_frame_results, hybrid_result

    def _apply_neg_threshold_to_results(self, frame_results: List) -> List:
        """
        Post-process frame results to apply neg threshold logic.
        Segments include trailing frames below neg_threshold at the end,
        but stop when probability rises above neg_threshold again or
        after min_silence_frames of below-threshold frames.
        """
        if not frame_results:
            return frame_results

        # Collect probabilities
        all_probs = [r.smoothed_prob for r in frame_results]

        # Get the min_speech_frame and min_silence_frame from the postprocessor
        min_speech_frames = self.vad.postprocessor.min_speech_frame
        min_silence_frames = self.vad.postprocessor.min_silence_frame

        console.print(
            f"[cyan]Neg threshold: {self.neg_threshold}, Min speech: {min_speech_frames} frames, Min silence: {min_silence_frames} frames[/cyan]"
        )

        # Find segments with trailing low-probability frames
        segments = []
        i = 0

        while i < len(all_probs):
            # Skip frames below threshold until we find speech
            if all_probs[i] < self.neg_threshold:
                i += 1
                continue

            # Found speech start
            segment_start = i

            # Find the main speech segment (frames above threshold)
            while i < len(all_probs) and all_probs[i] >= self.neg_threshold:
                i += 1

            # Now include trailing frames below threshold
            trailing_start = i
            trailing_count = 0

            while i < len(all_probs) and all_probs[i] < self.neg_threshold:
                trailing_count += 1
                i += 1

                # Stop if we've collected enough trailing frames
                if trailing_count >= min_silence_frames:
                    # Don't include this frame if we've reached min_silence
                    i -= 1  # Back up one frame
                    break

                # Stop if probability goes back above threshold
                # (This shouldn't happen since we check < threshold, but just in case)

            segment_end = i - 1

            segments.append((segment_start, segment_end))

            console.print(
                f"[dim]Segment: frames {segment_start}-{segment_end} "
                f"({segment_end - segment_start + 1} frames, "
                f"{(segment_end - segment_start + 1) * 0.01:.2f}s) "
                f"[trailing: {trailing_count} frames below threshold][/dim]"
            )

        console.print(
            f"[cyan]Raw segments with trailing frames: {len(segments)}[/cyan]"
        )

        # Apply min_speech_frame filter
        filtered_segments = []
        for start_idx, end_idx in segments:
            segment_duration = end_idx - start_idx + 1
            if segment_duration >= min_speech_frames:
                filtered_segments.append((start_idx, end_idx))
            else:
                console.print(
                    f"[yellow]Filtered out short segment: frames {start_idx}-{end_idx} "
                    f"({segment_duration} frames < {min_speech_frames})[/yellow]"
                )

        console.print(f"[cyan]After min_speech filter: {len(filtered_segments)}[/cyan]")

        # Apply min_silence_frame filter (merge close segments)
        if min_silence_frames > 0 and len(filtered_segments) > 1:
            merged_segments = [filtered_segments[0]]
            for current_start, current_end in filtered_segments[1:]:
                prev_start, prev_end = merged_segments[-1]
                silence_gap = current_start - prev_end

                if silence_gap <= min_silence_frames:
                    # Merge segments
                    merged_segments[-1] = (prev_start, current_end)
                    console.print(
                        f"[cyan]Merged segments with {silence_gap} frame gap[/cyan]"
                    )
                else:
                    merged_segments.append((current_start, current_end))

            final_segments = merged_segments
        else:
            final_segments = filtered_segments

        console.print(f"[green]Final segments: {len(final_segments)}[/green]")

        # Reset all segment markers
        for result in frame_results:
            try:
                result.is_speech_start = False
                result.is_speech_end = False
                if hasattr(result, "speech_start_frame"):
                    result.speech_start_frame = -1
                if hasattr(result, "speech_end_frame"):
                    result.speech_end_frame = -1
            except (AttributeError, TypeError):
                pass

        # Set new markers
        for start_idx, end_idx in final_segments:
            if 0 <= start_idx < len(frame_results):
                result = frame_results[start_idx]
                try:
                    result.is_speech_start = True
                    if hasattr(result, "speech_start_frame"):
                        result.speech_start_frame = start_idx + 1
                    console.print(
                        f"[green]Speech start: frame {start_idx} "
                        f"(prob={all_probs[start_idx]:.4f})[/green]"
                    )
                except (AttributeError, TypeError):
                    console.print(
                        f"[yellow]Warning: Could not set start marker on frame {start_idx}[/yellow]"
                    )

            if 0 <= end_idx < len(frame_results):
                result = frame_results[end_idx]
                try:
                    result.is_speech_end = True
                    if hasattr(result, "speech_end_frame"):
                        result.speech_end_frame = end_idx + 1
                    console.print(
                        f"[green]Speech end: frame {end_idx} "
                        f"(prob={all_probs[end_idx]:.4f})[/green]"
                    )
                except (AttributeError, TypeError):
                    console.print(
                        f"[yellow]Warning: Could not set end marker on frame {end_idx}[/yellow]"
                    )

        return frame_results

    def _apply_neg_threshold(self, prob: float) -> bool:
        """
        Check if current probability falls below neg_threshold.
        Returns True if segment should end, False otherwise.
        """
        if self._in_segment and prob < self.neg_threshold:
            console.print(
                f"[green]Segment ended due to neg_threshold: prob={prob:.4f} < {self.neg_threshold}[/green]"
            )
            self._in_segment = False
            self._current_segment_probs = []
            return True
        return False

    def _update_segment_state(self, prob: float) -> None:
        """Update segment tracking state based on probability."""
        if prob >= self.neg_threshold:
            if not self._in_segment:
                self._in_segment = True
                console.print(
                    f"[cyan]Segment started: prob={prob:.4f} >= {self.neg_threshold}[/cyan]"
                )
            self._current_segment_probs.append(prob)
        elif self._in_segment:
            console.print(
                f"[green]Segment ended: prob={prob:.4f} < {self.neg_threshold}[/green]"
            )
            self._in_segment = False
            self._current_segment_probs = []


def extract_speech_timestamps(
    audio: Union[str, Path, np.ndarray, torch.Tensor, list[np.ndarray]],
    threshold: float = DEFAULT_THRESHOLD,
    neg_threshold: float = DEFAULT_NEG_THRESHOLD,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_speech_duration_sec: float
    | None = DEFAULT_MAX_SPEECH_SEC,  # Now None by default
    return_seconds: bool = DEFAULT_RETURN_SECONDS,
    with_scores: bool = DEFAULT_WITH_SCORES,
    include_non_speech: bool = DEFAULT_INCLUDE_NON_SPEECH,
    smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
    pad_start_frame: int = DEFAULT_PAD_START_FRAME,
    max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
    use_hybrid: bool = DEFAULT_USE_HYBRID,
    **kwargs,
) -> Union[List[SpeechSegment], tuple[List[SpeechSegment], List[float]]]:
    """
    Extract speech timestamps using FireRedVAD.
    When include_non_speech=True, returns both speech and non-speech (silence) segments.

    Args:
        neg_threshold: Probability below which segments are ended (default: 0.1)
        max_speech_duration_sec: Maximum speech duration (None = unlimited)
    """
    # Remove the default assignment since it's now None by default
    # if max_speech_duration_sec is None:
    #     max_speech_duration_sec = DEFAULT_MAX_SPEECH_SEC

    audio_np, sr = load_audio(audio, sr=16000, mono=True)
    if sr != 16000:
        raise ValueError(f"FireRedVAD requires 16000 Hz, got {sr}")

    vad = get_global_vad(
        threshold=threshold,
        neg_threshold=neg_threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,  # Pass None directly
        smooth_window_size=smooth_window_size,
        pad_start_frame=pad_start_frame,
        max_buffer_sec=max_buffer_sec,
        use_hybrid=use_hybrid,
    )

    with console.status("[bold blue]Running FireRedVAD inference...[/bold blue]"):
        frame_results, result = vad.detect_full(audio_np)

    timestamps = result["timestamps"]
    probs = [r.smoothed_prob for r in frame_results]
    hop_sec = 0.010

    def make_segment(
        num: int,
        start_sec: float,
        end_sec: float,
        seg_type: Literal["speech", "non-speech"],
    ) -> SpeechSegment:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        frame_start = int(start_sec / hop_sec)
        frame_end = int(end_sec / hop_sec)
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

    if with_scores:
        return enhanced, probs
    return enhanced


def extract_speech_audio(
    audio: Union[str, Path, np.ndarray, torch.Tensor, list[np.ndarray]],
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    threshold: float = DEFAULT_THRESHOLD,
    neg_threshold: float = DEFAULT_NEG_THRESHOLD,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_speech_duration_sec: float
    | None = DEFAULT_MAX_SPEECH_SEC,  # Now None by default
    smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
    pad_start_frame: int = DEFAULT_PAD_START_FRAME,
    max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
    use_hybrid: bool = DEFAULT_USE_HYBRID,
) -> List[np.ndarray]:
    """
    Extract contiguous speech segments from the input audio using FireRedVAD.
    Returns a flat list of numpy arrays where each array represents one complete
    speech segment in float32 format, normalized to [-1.0, 1.0].
    """
    if sampling_rate != 16000:
        raise ValueError(f"FireRedVAD requires 16000 Hz, got {sampling_rate}")

    speech_segments = extract_speech_timestamps(
        audio=audio,
        threshold=threshold,
        neg_threshold=neg_threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,  # Can be None
        return_seconds=True,
        include_non_speech=False,
        smooth_window_size=smooth_window_size,
        pad_start_frame=pad_start_frame,
        max_buffer_sec=max_buffer_sec,
        use_hybrid=use_hybrid,
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
# Helpers used by save_segments
# ---------------------------------------------------------------------------


def _frames_from_seconds(sec: float) -> int:
    """Convert seconds to a 10 ms frame index (100 frames per second)."""
    return int(round(sec * 100.0))


def _compute_rms(
    signal: np.ndarray,
    frame_length: int = 160,
    hop_length: int = 160,
) -> np.ndarray:
    """
    Compute per-frame RMS energy aligned to 10 ms frames.
    160 samples @ 16 kHz = exactly 10 ms per frame.
    """
    if signal.size == 0:
        return np.array([], dtype=np.float32)
    num_frames = 1 + max(0, (len(signal) - frame_length) // hop_length)
    rms = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_length
        frame = signal[start : start + frame_length]
        if frame.size:
            rms[i] = float(np.sqrt(np.mean(frame**2)))
    return rms


def _generate_plot(
    probs: np.ndarray,
    segment_idx: int,
    duration_sec: float,
    output_path: Path,
    is_dummy: bool = False,
    rms: Optional[np.ndarray] = None,
) -> None:
    """Save a speech-probability (+ optional RMS energy) plot to *output_path*."""
    num_frames = len(probs)
    if num_frames == 0:
        return

    has_rms = rms is not None and len(rms) > 0
    rows = 2 if has_rms else 1
    fig, axes = plt.subplots(rows, 1, figsize=(9.5, 3.2 * rows), dpi=140)
    if rows == 1:
        axes = [axes]

    label = "Speech probability (dummy)" if is_dummy else "Speech probability"
    color = "#ff7f0e" if is_dummy else "#2ca02c"
    ax = axes[0]
    ax.plot(probs, color=color, linewidth=1.8, label=label)
    ax.fill_between(range(num_frames), probs, color=color, alpha=0.14)
    ax.axhline(
        y=0.4,
        linestyle="--",
        color="#d62728",
        alpha=0.65,
        linewidth=1.2,
        label="threshold ≈ 0.4",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, num_frames - 1)
    ax.set_ylabel("Speech Probability", fontsize=10.5)
    ax.set_xlabel(
        f"Frame (10 ms)  —  {num_frames} frames ≈ {duration_sec:.1f} s",
        fontsize=10.5,
    )
    ax.set_title(
        f"Segment {segment_idx:03d} — {'Dummy ' if is_dummy else ''}Model Probabilities",
        fontsize=12,
        pad=12,
    )
    ax.grid(True, alpha=0.28, linestyle="--", zorder=0)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    if has_rms:
        ax_rms = axes[1]
        ax_rms.plot(range(len(rms)), rms, linewidth=1.6, label="RMS energy")
        ax_rms.fill_between(range(len(rms)), rms, alpha=0.15)
        ax_rms.set_ylabel("RMS Energy", fontsize=10.5)
        ax_rms.set_xlabel("Frame (10 ms)", fontsize=10.5)
        ax_rms.set_xlim(0, len(rms) - 1)
        ax_rms.grid(True, alpha=0.28, linestyle="--", zorder=0)
        ax_rms.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    fig.tight_layout(pad=0.9)
    plt.savefig(output_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# save_segments
# ---------------------------------------------------------------------------


def save_segments(
    segments: List[SpeechSegment],
    audio_chunks: List[np.ndarray],
    output_base_dir: Path,
) -> List[SpeechSegment]:
    """
    Persist every speech segment to *output_base_dir/segments/segment_NNN/*.

    For each segment the function writes:
      sound.wav          – 16-kHz PCM-16 audio
      meta.json          – SpeechSegment metadata + probs_info summary
      speech_probs.json  – per-frame probabilities + summary stats
      energies.json      – per-frame RMS energy
      speech_and_rms.png – probability + RMS energy plot

    Parameters
    ----------
    segments:
        Output of ``extract_speech_timestamps(..., return_seconds=True,
        with_scores=True)``.  Non-speech segments are skipped automatically.
    audio_chunks:
        Output of ``extract_speech_audio()``.  Must contain one array per
        *speech* segment in the same order.
    output_base_dir:
        Root directory that will receive the ``segments/`` sub-tree.

    Returns
    -------
    List[SpeechSegment]
        Metadata for every saved segment (``output_path`` field populated).
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_base_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    speech_segments = [s for s in segments if s["type"] == "speech"]

    if len(speech_segments) != len(audio_chunks):
        console.print(
            f"[yellow]save_segments: {len(speech_segments)} speech segments but "
            f"{len(audio_chunks)} audio chunks — zipping by position, extras ignored.[/yellow]"
        )

    pairs = list(zip(speech_segments, audio_chunks))
    saved: List[SpeechSegment] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("[cyan]Saving segments + plots…", total=len(pairs))

        for meta, audio_np in pairs:
            idx = meta["num"]
            seg_dir = segments_dir / f"segment_{idx:03d}"
            seg_dir.mkdir(exist_ok=True)

            # ── 1. WAV ────────────────────────────────────────────────────
            wav_path = seg_dir / "sound.wav"
            try:
                sf.write(str(wav_path), audio_np, 16000, subtype="PCM_16")
            except Exception as exc:
                console.print(f"[red]Failed to save WAV {wav_path}: {exc}[/red]")
                progress.advance(task)
                continue

            # ── 2. Probability array ──────────────────────────────────────
            seg_probs_list: List[float] = meta.get("segment_probs", [])
            seg_probs_arr = np.asarray(seg_probs_list, dtype=np.float32)
            is_dummy = len(seg_probs_arr) == 0

            if is_dummy:
                # Synthetic sigmoid fallback so the plot is still meaningful
                num_frames = max(1, _frames_from_seconds(meta["duration"]))
                t = np.linspace(0, 1, num_frames)
                base = 0.12 + 0.76 / (1 + np.exp(-14 * (t - 0.48)))
                noise = np.random.default_rng().normal(0, 0.035, num_frames)
                seg_probs_arr = np.clip(base + noise, 0.03, 0.99).astype(np.float32)
                seg_probs_arr *= 0.88 + 0.12 * np.sin(np.pi * t) ** 0.35
                console.print(
                    f"[yellow]Segment {idx:03d}: no probabilities stored — "
                    "using synthetic fallback.[/yellow]"
                )

            # ── 3. probs_info summary stats ───────────────────────────────
            probs_info = {
                "num_frames": int(len(seg_probs_arr)),
                "mean": float(np.mean(seg_probs_arr)),
                "max": float(np.max(seg_probs_arr)),
                "min": float(np.min(seg_probs_arr)),
                "std": float(np.std(seg_probs_arr)),
                "median": float(np.median(seg_probs_arr)),
                "frame_rate_hz": 100,
            }

            # ── 4. meta.json ──────────────────────────────────────────────
            meta_to_save = dict(meta)
            meta_to_save["output_path"] = str(wav_path.relative_to(output_base_dir))
            meta_to_save["probs_info"] = probs_info
            # segment_probs can be large; keep it out of meta.json
            meta_to_save.pop("segment_probs", None)
            with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
                json.dump(meta_to_save, fh, indent=2, ensure_ascii=False)

            # ── 5. speech_probs.json ──────────────────────────────────────
            with open(seg_dir / "speech_probs.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "probs": seg_probs_arr.tolist(),
                        "frame_shift_sec": 0.010,
                        "frame_start": meta.get("frame_start", 0),
                        "summary": probs_info,
                        "is_dummy": is_dummy,
                    },
                    fh,
                    indent=2,
                )

            # ── 6. energies.json ──────────────────────────────────────────
            rms = _compute_rms(audio_np)
            with open(seg_dir / "energies.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "rms": rms.tolist(),
                        "frame_shift_sec": 0.010,
                        "num_frames": int(len(rms)),
                    },
                    fh,
                    indent=2,
                )

            # ── 7. speech_and_rms.png ─────────────────────────────────────
            _generate_plot(
                probs=seg_probs_arr,
                segment_idx=idx,
                duration_sec=float(meta["duration"]),
                output_path=seg_dir / "speech_and_rms.png",
                is_dummy=is_dummy,
                rms=rms,
            )

            meta["output_path"] = meta_to_save["output_path"]
            saved.append(meta)
            progress.advance(task)

    console.print(f"[bold green]✓ Saved {len(saved)} segments[/bold green]")
    console.print(
        f"Output: [link=file://{segments_dir.resolve()}]{segments_dir}[/link]"
    )
    return saved


if __name__ == "__main__":
    import argparse
    import platform
    import shutil
    import subprocess

    from jet.audio.speech.vad_extractors import load_probs

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    DEFAULT_AUDIO = str(
        Path("~/.cache/files/audio/recording_3_speakers.wav").expanduser().resolve()
    )

    parser = argparse.ArgumentParser(
        description="Extract speech segments with FireRedVAD"
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
        "-nt",
        "--neg-threshold",
        type=float,
        default=DEFAULT_NEG_THRESHOLD,
        help=f"threshold below which segments end (default: {DEFAULT_NEG_THRESHOLD})",
    )
    parser.add_argument(
        "-ms",
        "--min-silence",
        type=float,
        default=DEFAULT_MIN_SILENCE_SEC,
        help=f"minimum silence duration in seconds (default: {DEFAULT_MIN_SILENCE_SEC})",
    )
    parser.add_argument(
        "-mc",
        "--min-speech",
        type=float,
        default=DEFAULT_MIN_SPEECH_SEC,
        help=f"minimum speech duration in seconds (default: {DEFAULT_MIN_SPEECH_SEC})",
    )
    parser.add_argument(
        "-mx",
        "--max-speech",
        type=float,
        default=DEFAULT_MAX_SPEECH_SEC,
        help="maximum speech duration in seconds (None = unlimited)",
    )
    # === NEW ARGUMENTS ===
    parser.add_argument(
        "-mp",
        "--min-prob",
        type=float,
        default=0.0,
        help="minimum average speech probability to keep a segment (default: 0.0 = no filter)",
    )
    parser.add_argument(
        "-md",
        "--min-duration",
        type=float,
        default=0.0,
        help="minimum duration in seconds to keep a segment (default: 0.0 = no filter)",
    )
    parser.add_argument(
        "-sw",
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW_SIZE,
        help=f"smoothing window size (default: {DEFAULT_SMOOTH_WINDOW_SIZE})",
    )
    parser.add_argument(
        "-ps",
        "--pad-start",
        type=int,
        default=DEFAULT_PAD_START_FRAME,
        help=f"pad start frames (default: {DEFAULT_PAD_START_FRAME})",
    )
    parser.add_argument(
        "-mb",
        "--max-buffer-sec",
        type=float,
        default=DEFAULT_MAX_BUFFER_SEC,
        help=f"stream buffer duration in seconds (default: {DEFAULT_MAX_BUFFER_SEC})",
    )
    parser.add_argument(
        "--quantize",
        "-q",
        action="store_true",
        help="Quantize audio to int16 before processing (default: False)",
    )

    args = parser.parse_args()

    audio_path = args.audio_path
    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)

    console.rule("Audio Segmenter – FireRedVAD2", style="blue")
    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_path).name}\n")

    _, audio_np = load_probs(args.audio_path)

    if args.quantize:
        audio_np = convert_audio_dtype(audio_np, "int16")

    display_audio_info(audio_np)

    # ── Step 1: detect segments ───────────────────────────────────────────
    segments, speech_probs = extract_speech_timestamps(
        audio_np,
        threshold=args.threshold,
        neg_threshold=args.neg_threshold,
        min_silence_duration_sec=args.min_silence,
        min_speech_duration_sec=args.min_speech,
        max_speech_duration_sec=args.max_speech,  # Can be None
        return_seconds=True,
        with_scores=True,
        include_non_speech=False,
        smooth_window_size=args.smooth_window,
        pad_start_frame=args.pad_start,
        max_buffer_sec=args.max_buffer_sec,
    )

    # === NEW: Apply filters (min-prob and min-duration) ===
    original_count = len(segments)
    filtered = []

    for s in segments:
        if s.get("prob", 0.0) < args.min_prob:
            continue
        if s.get("duration", 0.0) < args.min_duration:
            continue
        filtered.append(s)

    segments = filtered

    if original_count != len(segments):
        console.print(
            f"[yellow]Filtered: {len(segments)}/{original_count} segments kept "
            f"(min-prob={args.min_prob:.3f}, min-duration={args.min_duration:.2f}s)[/yellow]"
        )

    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments)}\n")

    # ── Step 2: extract audio chunks for filtered segments ───────────────
    audio_chunks = extract_speech_audio(
        audio_path,
        sampling_rate=DEFAULT_SAMPLING_RATE,
        threshold=args.threshold,
        neg_threshold=args.neg_threshold,  # NEW
        min_silence_duration_sec=args.min_silence,
        min_speech_duration_sec=args.min_speech,
        max_speech_duration_sec=args.max_speech,
        smooth_window_size=args.smooth_window,
        pad_start_frame=args.pad_start,
        max_buffer_sec=args.max_buffer_sec,
    )

    # Safety: align audio chunks with filtered segments
    speech_segments = [s for s in segments if s["type"] == "speech"]
    audio_chunks = audio_chunks[: len(speech_segments)]

    # ── Step 3: save everything ───────────────────────────────────────────
    saved_metas = save_segments(segments, audio_chunks, output_dir)

    # Helper to play sound
    def play_segment(wav_path: Path):
        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["afplay", str(wav_path)], check=False)
            elif platform.system() == "Windows":
                subprocess.run(
                    [
                        "powershell",
                        "-c",
                        f"(New-Object Media.SoundPlayer '{wav_path}').PlaySync()",
                    ],
                    check=False,
                )
            else:  # Linux
                subprocess.run(["aplay", str(wav_path)], check=False)
        except Exception:
            pass  # silent fail

    # ── Step 4: display summary with Play buttons ────────────────────────
    for seg in saved_metas:
        seg_type = seg["type"]
        type_color = "bold green" if seg_type == "speech" else "bold red"
        wav_rel = seg.get("output_path")
        wav_full = output_dir / wav_rel if wav_rel else None

        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white]"
            f" - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"dur=[bold magenta]{seg['duration']:.2f}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan] "
            f"type=[{type_color}]{seg_type}[/{type_color}]"
            f"   [bold blue][link=file://{wav_full}]▶ Play[/link][/bold blue]"
        )

    if not any(s["type"] == "speech" for s in saved_metas):
        console.print("[red]No speech segments found after filtering.[/red]")
        raise SystemExit(0)

    # ── Step 5: write summary JSONs ───────────────────────────────────────
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

    console.rule("Done", style="green")
