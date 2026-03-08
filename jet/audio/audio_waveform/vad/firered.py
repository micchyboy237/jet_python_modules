from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class FireRedVADWrapper:
    """Streaming FireRedVAD wrapper"""

    def __init__(self, device: str | None = None) -> None:
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
            speech_threshold=0.65,
            smooth_window_size=5,
            pad_start_frame=4,
            min_speech_frame=6,
            max_speech_frame=2000,
            min_silence_frame=10,
            chunk_max_frame=30000,
        )

        self.vad = FireRedStreamVad.from_pretrained(model_dir, config=config)
        print("done.")

        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_prob = 0.0

    def get_speech_prob(self, chunk: np.ndarray) -> float:
        if len(chunk) == 0:
            return self.last_prob

        # Simple dynamic range compression / normalization
        chunk_max = np.max(np.abs(chunk)) + 1e-10
        target_peak = 0.30
        if chunk_max < 0.20:
            gain = min(target_peak / chunk_max, 8.0)
            chunk = chunk * gain
        elif chunk_max > 0.60:
            gain = 0.60 / chunk_max
            chunk = chunk * gain

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
