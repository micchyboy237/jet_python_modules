from typing import Union

import matplotlib
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.helpers.config import (
    FRAME_PER_SECONDS,
    SAMPLE_RATE,
)
from jet.audio.speech.vad_types import StreamVadFrame

matplotlib.use("Agg")
import numpy as np
import torch
from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig
from jet.audio.audio_waveform.vad.vad_config import (
    BUFFER_OVERLAP_SAMPLES,
    DEFAULT_MAX_BUFFER_SEC,
    DEFAULT_MAX_SPEECH_SEC,
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_MIN_SPEECH_SEC,
    DEFAULT_PROB_WEIGHT,
    DEFAULT_RMS_WEIGHT,
    DEFAULT_SMOOTH_WINDOW_SIZE,
    DEFAULT_THRESHOLD,
    MIN_BUFFER_SAMPLES_BEFORE_FIRST_VAD,
    SAVE_DIR,
    VAD_CONTEXT_WINDOW_SAMPLES,
)
from jet.audio.audio_waveform.vad.vad_hybrid_stream_vad_postprocessor import (
    HybridStreamVadPostprocessor,
)
from rich.console import Console

console = Console()

DEFAULT_PREROLL_FRAMES: int = 10  # 100 ms of look-back before detected start
DEFAULT_ONSET_FLOOR: float = 0.20  # walk back until prob drops below this

# ---------------------------------------------------------------------------
# FireRedVAD wrapper
# ---------------------------------------------------------------------------


class FireRedVAD:
    """Wrapper for FireRedVAD with simple streaming-like API."""

    def __init__(
        self,
        model_dir: str = SAVE_DIR,
        device: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
        min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
        max_speech_duration_sec: float = DEFAULT_MAX_SPEECH_SEC,
        smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
        max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        console.print(f"[cyan]Loading FireRedVAD (streaming) on {self.device}…[/cyan]")

        config = FireRedStreamVadConfig(
            use_gpu=(device == "cuda"),
            speech_threshold=threshold,
            smooth_window_size=smooth_window_size,
            min_speech_frame=int(min_speech_duration_sec * FRAME_PER_SECONDS),
            max_speech_frame=int(max_speech_duration_sec * FRAME_PER_SECONDS),
            min_silence_frame=int(min_silence_duration_sec * FRAME_PER_SECONDS),
            chunk_max_frame=30000,
        )

        self.vad = FireRedStreamVad.from_pretrained(model_dir, config=config)

        self.vad.postprocessor = HybridStreamVadPostprocessor(
            smooth_window_size=config.smooth_window_size,
            speech_threshold=config.speech_threshold,
            pad_start_frame=config.pad_start_frame,
            min_speech_frame=config.min_speech_frame,
            max_speech_frame=config.max_speech_frame,
            min_silence_frame=config.min_silence_frame,
            prob_weight=DEFAULT_PROB_WEIGHT,
            rms_weight=DEFAULT_RMS_WEIGHT,
        )

        self.vad.vad_model.to(self.device)
        console.print("[green]done.[/green]")

        self.sample_rate = SAMPLE_RATE
        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.last_prob: float = 0.0

        # Accumulated probabilities with full frame info
        self._accumulated_probs: list[StreamVadFrame] = []

        self.max_buffer_samples = int(max_buffer_sec * self.sample_rate)

    def reset(self) -> None:
        """Reset internal VAD state and clear audio buffer."""
        self.vad.reset()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_prob = 0.0
        self._accumulated_probs.clear()

    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Simple dynamic normalization"""
        chunk_max = np.max(np.abs(chunk)) + 1e-10
        target_peak = 0.30
        if chunk_max < 0.20:
            gain = min(target_peak / chunk_max, 8.0)
            return chunk * gain
        elif chunk_max > 0.60:
            gain = 0.60 / chunk_max
            return chunk * gain
        return chunk

    @torch.inference_mode()
    def get_speech_prob(self, chunk: np.ndarray) -> float:
        """
        Process incoming audio chunk and return the latest smoothed speech probability.
        Also accumulates full frame information including speech_start_frame and speech_end_frame.
        """
        if len(chunk) == 0:
            return self.last_prob

        chunk = self._normalize_chunk(chunk)
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])

        total_samples = len(self.audio_buffer)
        if total_samples < MIN_BUFFER_SAMPLES_BEFORE_FIRST_VAD:
            return self.last_prob

        full_buffer = self.audio_buffer
        to_process = full_buffer[-VAD_CONTEXT_WINDOW_SAMPLES:]

        try:
            results = self.vad.detect_chunk(to_process)

            # Keep only overlap for next iteration
            overlap = full_buffer[-BUFFER_OVERLAP_SAMPLES:]
            self.audio_buffer = overlap

            if not results:
                return self.last_prob

            # === Accumulate full frame results ===
            for res in results:
                self._accumulated_probs.append(
                    {
                        "frame_idx": res.frame_idx,
                        "raw_prob": float(res.raw_prob),
                        "smoothed_prob": float(res.smoothed_prob),
                        "is_speech": bool(res.is_speech),
                        "is_speech_start": bool(res.is_speech_start),
                        "is_speech_end": bool(res.is_speech_end),
                        "speech_start_frame": int(res.speech_start_frame),
                        "speech_end_frame": int(res.speech_end_frame),
                    }
                )

            last = results[-1]
            prob = last.smoothed_prob
            self.last_prob = prob
            return prob

        except Exception as e:
            console.print(f"[red]VAD detect_chunk error: {e}[/red]")
            self.audio_buffer = full_buffer[-BUFFER_OVERLAP_SAMPLES:]
            return self.last_prob

    def detect_full(
        self,
        audio: Union[str, np.ndarray],
    ) -> tuple[list, dict]:
        """Full-file detection with accumulated probabilities."""
        self.reset()
        frame_results, result = self.vad.detect_full(audio)

        # Populate accumulator with full information
        self._accumulated_probs = [
            {
                "frame_idx": r.frame_idx,
                "raw_prob": float(r.raw_prob),
                "smoothed_prob": float(r.smoothed_prob),
                "is_speech": bool(r.is_speech),
                "is_speech_start": bool(r.is_speech_start),
                "is_speech_end": bool(r.is_speech_end),
                "speech_start_frame": int(r.speech_start_frame),
                "speech_end_frame": int(r.speech_end_frame),
            }
            for r in frame_results
        ]

        return frame_results, result

    def get_accumulated_probs(self) -> list[StreamVadFrame]:
        """Return copy of all accumulated frame probabilities (including start/end frames)."""
        return self._accumulated_probs.copy()

    def get_prob_history(self, as_numpy: bool = False):
        """Convenience method for numpy arrays (frames + probabilities)."""
        if not self._accumulated_probs:
            if as_numpy:
                return np.array([], dtype=int), np.array([], dtype=float)
            return []

        if not as_numpy:
            return self.get_accumulated_probs()

        frames = np.array([p["frame_idx"] for p in self._accumulated_probs], dtype=int)
        smoothed = np.array(
            [p["smoothed_prob"] for p in self._accumulated_probs], dtype=float
        )
        return frames, smoothed

    def get_speech_segments(
        self,
        return_seconds: bool = False,
        audio_np: np.ndarray | None = None,
        preroll_frames: int = DEFAULT_PREROLL_FRAMES,
        onset_floor: float = DEFAULT_ONSET_FLOOR,
    ) -> tuple[list[SpeechSegment], list[np.ndarray]]:
        """
        Parse _accumulated_probs into a list of SpeechSegment using make_segment(),
        and optionally slice the corresponding audio chunks from audio_np.

        Args:
            return_seconds:  If True, segment start/end are in seconds (float).
                            If False (default), they are in samples (int).
            audio_np:        Optional full audio array. When provided, each segment's
                            audio slice is returned as the second element of the tuple.
            preroll_frames:  Number of frames to look back before the detected
                            speech_start_frame to capture the onset ramp-up.
            onset_floor:     Walk the pre-roll back further until smoothed_prob
                            drops below this value, preventing mid-word starts.
                            Set to 0.0 to use a fixed preroll_frames only.
        Returns:
            A tuple of:
            - list[SpeechSegment]: detected speech segments
            - list[np.ndarray]:    corresponding audio chunks (empty list if audio_np is None)
        """
        from jet.audio.audio_waveform.vad.vad_utils import make_segment
        from jet.audio.helpers.config import FRAME_PER_SECONDS

        # Build frame_idx → list position index for O(1) look-back
        frame_idx_to_pos: dict[int, int] = {
            f["frame_idx"]: i for i, f in enumerate(self._accumulated_probs)
        }

        segments: list[SpeechSegment] = []
        seg_num: int = 0
        start_frame: int = -1
        segment_probs: list[float] = []
        prev_end_frame: int = 0  # track where the last segment ended to prevent overlap

        for frame in self._accumulated_probs:
            if frame["is_speech_start"]:
                raw_start_frame = frame["speech_start_frame"]
                earliest_pos = frame_idx_to_pos.get(raw_start_frame, 0)

                # Pre-roll: walk back up to preroll_frames before the detected start
                preroll_pos = max(0, earliest_pos - preroll_frames)

                # Further walk back while prob is still above onset_floor
                if onset_floor > 0.0:
                    while preroll_pos > 0:
                        if (
                            self._accumulated_probs[preroll_pos]["smoothed_prob"]
                            < onset_floor
                        ):
                            break
                        preroll_pos -= 1

                # Overlap guard: never start before the previous segment ended
                if prev_end_frame > 0:
                    prev_end_pos = frame_idx_to_pos.get(prev_end_frame, 0)
                    preroll_pos = max(preroll_pos, prev_end_pos + 1)

                start_frame = self._accumulated_probs[preroll_pos]["frame_idx"]
                segment_probs = [
                    float(f["smoothed_prob"])
                    for f in self._accumulated_probs[preroll_pos:earliest_pos]
                ]

            if start_frame != -1 and not frame["is_speech_start"]:
                segment_probs.append(float(frame["smoothed_prob"]))

            if frame["is_speech_end"] and start_frame != -1:
                end_frame = frame["speech_end_frame"]
                prev_end_frame = end_frame  # record for overlap guard on next segment

                start_sec = max(0.0, (start_frame - 1) / FRAME_PER_SECONDS)
                end_sec = max(0.0, (end_frame - 1) / FRAME_PER_SECONDS)
                seg = make_segment(
                    num=seg_num,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    probs=segment_probs or [0.0],
                    seg_type="speech",
                    is_ongoing=False,
                    return_seconds=return_seconds,
                )
                segments.append(seg)
                seg_num += 1
                start_frame = -1
                segment_probs = []

        # Handle ongoing segment at end of audio
        if start_frame != -1 and self._accumulated_probs:
            last_frame_idx = self._accumulated_probs[-1]["frame_idx"]
            start_sec = max(0.0, (start_frame - 1) / FRAME_PER_SECONDS)
            end_sec = max(0.0, (last_frame_idx - 1) / FRAME_PER_SECONDS)
            seg = make_segment(
                num=seg_num,
                start_sec=start_sec,
                end_sec=end_sec,
                probs=segment_probs or [0.0],
                seg_type="speech",
                is_ongoing=True,
                return_seconds=return_seconds,
            )
            segments.append(seg)

        audio_chunks: list[np.ndarray] = []
        if audio_np is not None:
            audio_chunks = [
                audio_np[int(seg["start"]) : int(seg["end"])] for seg in segments
            ]

        return segments, audio_chunks
