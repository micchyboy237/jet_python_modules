from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
from fireredvad.core.audio_feat import AudioFeat
from fireredvad.core.detect_model import DetectModel
from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig
from jet.audio.audio_waveform.hybrid_stream_vad_postprocessor import (
    HybridStreamVadPostprocessor,
)
from jet.audio.audio_waveform.speech_segment_tracker import SpeechSegmentTracker
from jet.audio.helpers.energy_base import compute_rms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger("fireredvad.bin.stream_vad")


# ────────────────────────────────────────────────
# Updated Streaming Constants (aligned with FireRedVAD)
# ────────────────────────────────────────────────
MIN_BUFFER_SAMPLES_BEFORE_FIRST_VAD = 4800  # ~300 ms

# Context window: multiple of frame shift for clean processing
VAD_CONTEXT_WINDOW_SAMPLES = 9600  # 600 ms (60 frames)

# Overlap must be multiple of FRAME_SHIFT_SAMPLE (160)
BUFFER_OVERLAP_SAMPLES = 640  # 40 frames (~40 ms) — good trade-off
# ────────────────────────────────────────────────


class FireRedVADWrapper:
    """Streaming FireRedVAD wrapper — optimized"""

    def __init__(
        self,
        tracker: SpeechSegmentTracker | None = None,
        device: str | None = None,
        smooth_window_size: int = 5,
        speech_threshold: float = 0.5,
        pad_start_frame: int = 5,
        min_speech_frame: int = 30,
        soft_max_speech_frame: int = 450,
        hard_max_speech_frame: int = 800,
        min_silence_frame: int = 20,
        chunk_max_frame: int = 30000,
        search_window: int = 200,
        valley_threshold: float = 0.65,
        min_valley_consecutive_frames: int = 5,
    ) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        print(f"Loading FireRedVAD **streaming** on {device}... ", end="", flush=True)

        model_dir = str(
            Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD")
            .expanduser()
            .resolve()
        )

        config = FireRedStreamVadConfig(
            use_gpu=(device == "cuda"),
            speech_threshold=speech_threshold,
            smooth_window_size=smooth_window_size,
            pad_start_frame=pad_start_frame,
            min_speech_frame=min_speech_frame,
            max_speech_frame=hard_max_speech_frame,
            min_silence_frame=min_silence_frame,
            chunk_max_frame=chunk_max_frame,
        )

        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        feat_extractor = AudioFeat(cmvn_path)

        vad_model = DetectModel.from_pretrained(model_dir)
        if config.use_gpu:
            vad_model.cuda()
        else:
            vad_model.cpu()

        postprocessor = HybridStreamVadPostprocessor(
            smooth_window_size=smooth_window_size,
            speech_threshold=speech_threshold,
            pad_start_frame=pad_start_frame,
            min_speech_frame=min_speech_frame,
            soft_max_speech_frame=soft_max_speech_frame,
            hard_max_speech_frame=hard_max_speech_frame,
            min_silence_frame=min_silence_frame,
            search_window=search_window,
            valley_threshold=valley_threshold,
            min_valley_consecutive_frames=min_valley_consecutive_frames,
        )

        self.vad = FireRedStreamVad(
            audio_feat=feat_extractor,
            vad_model=vad_model,
            postprocessor=postprocessor,
            config=config,
        )

        # Rolling buffer (list of arrays is efficient for this pattern)
        self.audio_buffer: list[np.ndarray] = []
        self.last_prob = 0.0
        self._peak_rms: float = 1e-6

        self.tracker = tracker or SpeechSegmentTracker()
        self.tracker.postprocessor = self.vad.postprocessor

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

    def get_speech_prob(self, chunk: np.ndarray) -> float:
        if len(chunk) == 0:
            return self.last_prob

        chunk = self._normalize_chunk(chunk)
        self.audio_buffer.append(chunk)

        chunk_rms = compute_rms(chunk)
        if chunk_rms > self._peak_rms:
            self._peak_rms = chunk_rms
        norm_rms = min(chunk_rms / self._peak_rms, 1.0)

        # Early exit if not enough audio yet
        total_samples = sum(len(c) for c in self.audio_buffer)
        if total_samples < MIN_BUFFER_SAMPLES_BEFORE_FIRST_VAD:
            return self.last_prob

        # Concatenate only when processing
        full_buffer = np.concatenate(self.audio_buffer)
        to_process = full_buffer[-VAD_CONTEXT_WINDOW_SAMPLES:]

        try:
            results = self.vad.detect_chunk(to_process)

            # Keep only the overlap portion for next iteration
            overlap = full_buffer[-BUFFER_OVERLAP_SAMPLES:]
            self.audio_buffer = [overlap]

            if not results:
                return self.last_prob

            last = results[-1]
            prob = last.smoothed_prob
            self.last_prob = prob

            # Feed to tracker
            if self.tracker is not None:
                for result in results:
                    # smoothed_prob comes from the VAD model (speech probability, 0–1).
                    # norm_rms is the energy signal we computed above (0–1).
                    # hybrid blends them 50/50 as per the formula:
                    # hybrid[i] = 0.5 × smoothed_prob[i] + 0.5 × norm_rms[i]
                    hybrid_prob = 0.5 * result.smoothed_prob + 0.5 * norm_rms
                    self.tracker.on_frame(
                        result,
                        rms=chunk_rms,
                        hybrid_prob=round(hybrid_prob, 4),
                    )

            return prob

        except Exception as e:
            logger.error(f"VAD detect_chunk error: {e}")
            # Fallback: keep last prob and trim buffer
            self.audio_buffer = [full_buffer[-BUFFER_OVERLAP_SAMPLES:]]
            return self.last_prob
