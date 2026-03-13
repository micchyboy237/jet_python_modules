from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger("fireredvad.bin.stream_vad")


# ────────────────────────────────────────────────
# Streaming / buffer constants (all assuming 16 kHz sample rate)
# ────────────────────────────────────────────────
SAMPLE_RATE = 16000

MIN_BUFFER_SAMPLES_BEFORE_FIRST_VAD = 4800  # ≈ 300 ms
VAD_CONTEXT_WINDOW_SAMPLES = 9600  # ≈ 600 ms
BUFFER_OVERLAP_SAMPLES = 1600  # ≈ 100 ms

# Default long history = 5 minutes
DEFAULT_HISTORY_SECONDS = 300
DEFAULT_MAX_HISTORY_SAMPLES = DEFAULT_HISTORY_SECONDS * SAMPLE_RATE
# ────────────────────────────────────────────────


class FireRedVADWrapper:
    """Streaming FireRedVAD wrapper with separate long audio history buffer"""

    def __init__(
        self,
        device: str | None = None,
        max_history_seconds: float = DEFAULT_HISTORY_SECONDS,
    ) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        print(f"Loading FireRedVAD **streaming** on {device}... ", end="", flush=True)

        from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig

        model_dir = str(
            Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD")
            .expanduser()
            .resolve()
        )

        config = FireRedStreamVadConfig(
            use_gpu=(device == "cuda"),
            speech_threshold=0.5,
            smooth_window_size=5,
            pad_start_frame=5,
            min_speech_frame=30,
            max_speech_frame=500,
            min_silence_frame=20,
            chunk_max_frame=30000,
        )

        self.vad = FireRedStreamVad.from_pretrained(model_dir, config=config)
        print("done.")

        # ─── Short buffer: only for VAD continuity / overlap ───
        self.audio_buffer = np.array([], dtype=np.float32)

        # ─── Long history buffer: keeps last N minutes ─────────
        self.max_history_samples = int(max_history_seconds * SAMPLE_RATE)
        self.audio_history = np.array([], dtype=np.float32)

        self.last_prob = 0.0

    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk_max = np.max(np.abs(chunk)) + 1e-10
        target_peak = 0.30

        if chunk_max < 0.20:
            gain = min(target_peak / chunk_max, 8.0)
            chunk = chunk * gain
        elif chunk_max > 0.60:
            gain = 0.60 / chunk_max
            chunk = chunk * gain

        return chunk

    def _update_history(self, chunk: np.ndarray) -> None:
        """Append new audio to the long history and trim if necessary"""
        if len(chunk) == 0:
            return

        self.audio_history = np.concatenate([self.audio_history, chunk])

        # Keep only the most recent part (sliding window)
        if len(self.audio_history) > self.max_history_samples:
            self.audio_history = self.audio_history[-self.max_history_samples :]

    def get_speech_prob(self, chunk: np.ndarray) -> float:
        """
        Main entry point.
        Returns speech probability (0–1) for the latest context.
        """
        if len(chunk) == 0:
            return self.last_prob

        # Normalize → update short buffer → update long history
        chunk_norm = self._normalize_chunk(chunk)
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk_norm])
        self._update_history(chunk_norm)  # ← long buffer updated here

        # Not enough data yet → return previous value
        if len(self.audio_buffer) < MIN_BUFFER_SAMPLES_BEFORE_FIRST_VAD:
            return self.last_prob

        # Feed only the most recent context window to VAD
        to_process = self.audio_buffer[-VAD_CONTEXT_WINDOW_SAMPLES:]
        results = self.vad.detect_chunk(to_process)

        # Keep only overlap for next call (smooth transitions)
        self.audio_buffer = self.audio_buffer[-BUFFER_OVERLAP_SAMPLES:]

        if not results:
            return self.last_prob

        # Take the most recent smoothed probability
        last = results[-1]
        prob = last.smoothed_prob
        self.last_prob = prob

        return prob

    # ─── Convenience / debug methods ─────────────────────────────────────

    def get_history_duration_seconds(self) -> float:
        return len(self.audio_history) / SAMPLE_RATE

    def get_history_length_samples(self) -> int:
        return len(self.audio_history)

    def clear_history(self) -> None:
        """Reset long audio history (but keep VAD state)"""
        self.audio_history = np.array([], dtype=np.float32)

    def get_recent_history(self, seconds: float) -> np.ndarray:
        """Get up to `seconds` of most recent audio from history"""
        samples = int(seconds * SAMPLE_RATE)
        if len(self.audio_history) <= samples:
            return self.audio_history.copy()
        return self.audio_history[-samples:]

    def get_all_history(self) -> np.ndarray:
        """Return copy of complete current history buffer"""
        return self.audio_history.copy()
