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
# Streaming / buffer constants (all assuming 16 kHz sample rate)
# ────────────────────────────────────────────────
MIN_BUFFER_SAMPLES_BEFORE_FIRST_VAD = (
    4800  # ≈ 300 ms – don't run VAD until we have at least this much audio
)
VAD_CONTEXT_WINDOW_SAMPLES = (
    9600  # ≈ 600 ms – how much recent audio to feed the model each time
)
BUFFER_OVERLAP_SAMPLES = (
    512  # ≈  32 ms – how much audio to keep for smooth continuity / next call
)
# ────────────────────────────────────────────────


class FireRedVADWrapper:
    """Streaming FireRedVAD wrapper"""

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

        # Use list buffer to avoid repeated concatenate
        self.vad = FireRedStreamVad(
            audio_feat=feat_extractor,
            vad_model=vad_model,
            postprocessor=postprocessor,
            config=config,
        )
        # self.vad = FireRedStreamVad.from_pretrained(model_dir, config=config)

        self.audio_chunks: list[np.ndarray] = []
        self.last_prob = 0.0

        self.tracker = tracker or SpeechSegmentTracker()
        # Give tracker access to postprocessor so it can read forced_split etc.
        self.tracker.postprocessor = self.vad.postprocessor

        # Rolling peak RMS used to normalize energy into [0, 1].
        # Starts at a small positive value so the first frame never divides by zero.
        # It grows whenever a louder chunk arrives, ensuring norm_rms stays in [0, 1].
        self._peak_rms: float = 1e-6

    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        # Simple dynamic range compression / normalization
        chunk_max = np.max(np.abs(chunk)) + 1e-10
        target_peak = 0.30
        if chunk_max < 0.20:
            gain = min(target_peak / chunk_max, 8.0)
            chunk = chunk * gain
        elif chunk_max > 0.60:
            gain = 0.60 / chunk_max
            chunk = chunk * gain
        return chunk

    def get_speech_prob(self, chunk: np.ndarray) -> float:
        if len(chunk) == 0:
            return self.last_prob

        chunk = self._normalize_chunk(chunk)
        self.audio_chunks.append(chunk)

        # ── RMS of this single chunk ──────────────────────────────────────────
        # compute_rms returns Root-Mean-Square energy (0 = silence, ~0.7 = loud).
        chunk_rms = compute_rms(chunk)

        # Keep a running maximum so we can scale any chunk's RMS to [0, 1].
        # Using max() means the scale never shrinks mid-stream, which would
        # cause earlier frames to look artificially louder.
        if chunk_rms > self._peak_rms:
            self._peak_rms = chunk_rms

        # norm_rms: how loud is this chunk relative to the loudest chunk seen so far?
        # Result is always in [0, 1].
        norm_rms = min(chunk_rms / self._peak_rms, 1.0)

        total_len = sum(len(c) for c in self.audio_chunks)

        if total_len < MIN_BUFFER_SAMPLES_BEFORE_FIRST_VAD:
            return self.last_prob

        # Concatenate only when needed
        audio_buffer = np.concatenate(self.audio_chunks)
        to_process = audio_buffer[-VAD_CONTEXT_WINDOW_SAMPLES:]
        results = self.vad.detect_chunk(to_process)

        # Keep only overlap (convert back to list)
        overlap = audio_buffer[-BUFFER_OVERLAP_SAMPLES:]
        self.audio_chunks = [overlap]

        if not results:
            return self.last_prob

        last = results[-1]
        prob = last.smoothed_prob

        self.last_prob = prob
        if self.tracker is not None:
            for result in results:
                # smoothed_prob comes from the VAD model (speech probability, 0–1).
                # norm_rms is the energy signal we computed above (0–1).
                # hybrid blends them 50/50 as per the formula:
                #   hybrid[i] = 0.5 × smoothed_prob[i] + 0.5 × norm_rms[i]
                hybrid_prob = 0.5 * result.smoothed_prob + 0.5 * norm_rms
                self.tracker.on_frame(
                    result,
                    rms=chunk_rms,
                    hybrid_prob=round(hybrid_prob, 4),
                )
            # self.tracker.add_prob(prob)
            # NOTE: tracker.add_prob() is intentionally NOT called here.
            # It is called once per frame in coordinated_callback() in
            # start_manager.py, using whichever VAD the user has selected
            # in the dropdown.  Calling it here too would double-count
            # FireRed's probability and bypass the VAD selector.
            # The on_frame() calls above are kept because only FireRed
            # generates the segment boundary events (speech_start / speech_end).

        return prob
