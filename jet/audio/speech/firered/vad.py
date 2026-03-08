from typing import Optional, Union

import numpy as np
import torch
from fireredvad.core.constants import FRAME_LENGTH_SAMPLE, SAMPLE_RATE
from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig

# assuming SAVE_DIR is defined somewhere; adjust if needed
from jet.audio.speech.firered.config import SAVE_DIR


class FireRedVAD:
    """Wrapper for FireRedVAD with simple streaming-like API."""

    def __init__(
        self,
        model_dir: str = SAVE_DIR,
        device: str | None = None,
        threshold: float = 0.65,
        min_silence_duration_sec: float = 0.20,  # 200 ms
        min_speech_duration_sec: float = 0.15,  # 150 ms
        max_speech_duration_sec: float = 12.0,
        smooth_window_size: int = 5,
        pad_start_frame: int = 5,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        print(
            f"Loading FireRedVAD (streaming) on {self.device}... ", end="", flush=True
        )

        # Convert durations → frame counts (100 frames = 1 second)
        frames_per_sec = 100

        config = FireRedStreamVadConfig(
            use_gpu=(device == "cuda"),
            speech_threshold=threshold,
            smooth_window_size=smooth_window_size,
            pad_start_frame=pad_start_frame,
            min_speech_frame=int(min_speech_duration_sec * frames_per_sec),
            max_speech_frame=int(max_speech_duration_sec * frames_per_sec),
            min_silence_frame=int(min_silence_duration_sec * frames_per_sec),
            chunk_max_frame=30000,
        )

        self.vad = FireRedStreamVad.from_pretrained(model_dir, config=config)
        self.vad.vad_model.to(self.device)
        print("done.")

        self.sample_rate = SAMPLE_RATE  # 16000
        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.last_prob: float = 0.0

        # Minimal look-back — just enough for model right-context + smoothing
        self.max_buffer_samples = int(1.2 * self.sample_rate)  # ~1.2 seconds max

    def reset(self) -> None:
        """Reset internal VAD state and clear audio buffer."""
        self.vad.reset()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_prob = 0.0

    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Simple dynamic range compression / gain normalization"""
        if len(chunk) == 0:
            return chunk.astype(np.float32)

        chunk = chunk.astype(np.float32)
        peak = np.max(np.abs(chunk)) + 1e-10

        target_peak = 0.30
        if peak < 0.18:
            gain = min(target_peak / peak, 9.0)
            chunk *= gain
        elif peak > 0.70:
            gain = 0.65 / peak
            chunk *= gain

        return chunk

    @torch.inference_mode()
    def get_speech_prob(self, chunk: np.ndarray) -> float:
        """
        Process incoming audio chunk (any length) and return
        the **latest smoothed speech probability**.
        """
        if len(chunk) == 0:
            return self.last_prob

        # Normalize level (helps a lot with real mic input)
        chunk = self._normalize_chunk(chunk)

        # Append new audio
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])

        # Trim very old history (keep memory usage bounded)
        if len(self.audio_buffer) > self.max_buffer_samples:
            self.audio_buffer = self.audio_buffer[-self.max_buffer_samples :]

        # We need at least ~0.3–0.5 s of audio to get meaningful results
        min_samples_for_inference = 4800  # 0.3 seconds
        if len(self.audio_buffer) < min_samples_for_inference:
            return self.last_prob

        # Process the recent part
        to_process = self.audio_buffer.copy()  # avoid mutating while processing

        results = self.vad.detect_chunk(to_process)

        if not results:
            return self.last_prob

        # Take the most recent frame's smoothed probability
        latest = results[-1]
        prob = latest.smoothed_prob
        self.last_prob = prob

        # Keep a small tail for continuity (helps model context)
        # ~32–64 ms is usually enough
        tail_samples = FRAME_LENGTH_SAMPLE * 6  # ~180 ms
        self.audio_buffer = self.audio_buffer[-tail_samples:]

        return prob

    def get_latest_result(self) -> Optional[dict]:
        """
        Optional: return more detailed info about the last processed frame
        (useful for debugging or when you need is_speech_start / is_speech_end)
        """
        # This would require keeping the last result object — omitted for simplicity
        # You can extend the class to store self.last_result = results[-1] if needed
        return None

    # Optional: full-file processing (unchanged from original)
    def detect_full(
        self,
        audio: Union[str, np.ndarray],
    ) -> tuple[list, dict]:
        self.reset()
        frame_results, result = self.vad.detect_full(audio)
        return frame_results, result
