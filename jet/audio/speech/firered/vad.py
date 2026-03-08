from typing import Union

import numpy as np
import torch
from fireredvad.core.constants import FRAME_LENGTH_SAMPLE
from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig
from jet.audio.speech.firered.config import SAVE_DIR


class FireRedVAD:
    """Wrapper for FireRedVAD with simple streaming-like API."""

    def __init__(
        self,
        model_dir: str = SAVE_DIR,
        device: str | None = None,
        threshold: float = 0.5,
        min_silence_duration_sec: float = 0.250,
        min_speech_duration_sec: float = 0.250,
        max_speech_duration_sec: float = 15.0,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(
            f"Loading FireRedVAD on {self.device}... ",
            end="",
            flush=True,
        )
        config = FireRedStreamVadConfig(
            use_gpu=(device == "cuda"),
            speech_threshold=threshold,
            min_speech_frame=int(min_speech_duration_sec * 100),  # 100 frames/sec
            min_silence_frame=int(min_silence_duration_sec * 100),
            max_speech_frame=int(max_speech_duration_sec * 100),
            pad_start_frame=5,
            smooth_window_size=5,
            chunk_max_frame=30000,
        )
        self.vad = FireRedStreamVad.from_pretrained(model_dir, config)
        self.vad.vad_model.to(self.device)
        print("done.")
        self.sample_rate = 16000
        self.context_samples = int(0.5 * self.sample_rate)  # 0.5 seconds context
        self.audio_ring: np.ndarray = np.zeros(self.context_samples, dtype=np.float32)
        self.write_pos = 0
        self.frame_buffer = np.zeros(FRAME_LENGTH_SAMPLE, dtype=np.float32)
        self.frame_samples_collected = 0

    def reset(self) -> None:
        """Reset VAD state and clear buffers."""
        self.vad.reset()
        self.audio_ring.fill(0.0)
        self.write_pos = 0
        self.frame_buffer.fill(0.0)
        self.frame_samples_collected = 0

    def get_speech_prob(self, chunk: np.ndarray) -> float:
        """Process an audio chunk and return speech probability for the latest frame."""
        if len(chunk) == 0:
            return 0.0
        chunk = chunk.astype(np.float32)
        # Update context buffer
        space = self.context_samples - self.write_pos
        if len(chunk) <= space:
            self.audio_ring[self.write_pos : self.write_pos + len(chunk)] = chunk
            self.write_pos += len(chunk)
        else:
            self.audio_ring[:] = chunk[-self.context_samples :]
            self.write_pos = self.context_samples

        # Process chunk in frame-sized pieces (400 samples = 25ms at 16000 Hz)
        prob = 0.0
        samples = chunk
        while len(samples) > 0:
            samples_needed = FRAME_LENGTH_SAMPLE - self.frame_samples_collected
            samples_to_take = min(samples_needed, len(samples))
            self.frame_buffer[
                self.frame_samples_collected : self.frame_samples_collected
                + samples_to_take
            ] = samples[:samples_to_take]
            self.frame_samples_collected += samples_to_take
            samples = samples[samples_to_take:]

            if self.frame_samples_collected == FRAME_LENGTH_SAMPLE:
                # Process full frame
                frame_result = self.vad.detect_frame(self.frame_buffer)
                prob = frame_result.smoothed_prob
                self.frame_buffer.fill(0.0)
                self.frame_samples_collected = 0

        return prob

    def detect_full(
        self,
        audio: Union[str, np.ndarray],
    ) -> tuple[list, dict]:
        """Process full audio and return frame results and timestamps."""
        self.reset()
        frame_results, result = self.vad.detect_full(audio)
        return frame_results, result
