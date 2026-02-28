# jet_python_modules/jet/audio/speech/speechbrain/vad.py

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from jet.audio.speech.speechbrain.speech_timestamps_extractor import (
    extract_speech_timestamps,
)
from jet.audio.speech.speechbrain.speech_types import SpeechSegment

VAD_CHUNK_SAMPLES: int = 512


@dataclass(frozen=True)
class Config:
    vad_chunk_bytes: float = VAD_CHUNK_SAMPLES * 2
    chunk_duration_sec: float = 6.0
    min_speech_duration_sec: float = 0.25
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    vad_start_threshold: float = 0.5  # hysteresis start
    vad_end_threshold: float = 0.25  # hysteresis end
    pre_roll_seconds: float = 0.35  # capture mora onsets
    max_silence_seconds: float = 0.9  # JP clause pauses
    vad_model_path: str | None = None  # allow custom model if needed
    max_rtt_history: int = 10
    reconnect_attempts: int = 5
    reconnect_delay: float = 3.0
    max_context_duration_sec: float = 90.0


config = Config()


class SpeechTracker:
    def __init__(self) -> None:
        self.audio_buffer = bytearray()
        self.pre_roll_buffer: deque[bytes] = deque(
            maxlen=int(
                config.pre_roll_seconds * config.sample_rate / config.vad_chunk_bytes
            )
        )

    def add_audio_chunk(self, chunk: bytes) -> float:
        self.audio_buffer.extend(chunk)


class SpeechManager:
    def __init__(self) -> None:
        self.processed_segments: list[SpeechSegment] = []
        self.active_segment: Optional[SpeechSegment] = None

    def process_buffer(self, audio_buffer: bytearray):
        audio_np = np.frombuffer(audio_buffer, dtype=np.int16).copy()
        segments, all_speech_probs = extract_speech_timestamps(
            audio_np,
            # min_speech_duration_ms=min_speech_duration_ms,
            # min_silence_duration_ms=min_silence_duration_ms,
            max_speech_duration_sec=config.chunk_duration_sec,
            return_seconds=True,
            time_resolution=3,
            with_scores=True,
            normalize_loudness=False,
            include_non_speech=True,
            double_check=True,
        )

        last_segment, complete_segments = self._split_segments(segments)


    def _split_segments(
        self, segments: list[SpeechSegment]
    ) -> tuple[SpeechSegment | None, list[SpeechSegment]]:
        last_segment = segments[-1] if segments else None
        complete_segments = [s for s in segments[:-1] if s["type"] == "speech"]

        return last_segment, complete_segments

    def _add_complete_segments_to_queue():
