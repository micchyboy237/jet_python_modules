import logging
import queue
from dataclasses import dataclass
from typing import Generator, List, Optional, TypedDict

import numpy as np
import sounddevice as sd
from fireredvad.core.constants import (
    FRAME_LENGTH_SAMPLE,
    FRAME_SHIFT_SAMPLE,
    SAMPLE_RATE,
)
from fireredvad.stream_vad import FireRedStreamVad

logger = logging.getLogger(__name__)


class SpeechSegment(TypedDict):
    start_time: float
    end_time: float
    audio: np.ndarray


@dataclass
class StreamingSpeechTrackerConfig:
    samplerate: int = SAMPLE_RATE
    device: Optional[int | str] = None
    dtype: str = "float32"
    channels: int = 1
    blocksize: int = FRAME_SHIFT_SAMPLE


class StreamingSpeechTracker:
    """
    Streaming speech segment detector using FireRed VAD.

    Responsibilities:
    - read audio from sounddevice
    - feed frames into FireRed VAD
    - accumulate speech segments
    - yield segments as soon as they end
    """

    def __init__(
        self,
        vad: FireRedStreamVad,
        config: StreamingSpeechTrackerConfig = StreamingSpeechTrackerConfig(),
    ):
        self.vad = vad
        self.config = config

        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._running = False

        self._frame_buffer = np.zeros(0, dtype=np.float32)

        self._current_segment_audio: List[np.ndarray] = []
        self._current_start_time: Optional[float] = None

        self._frame_index = 0

    # -----------------------------------------------------
    # Audio callback
    # -----------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)

        audio = indata[:, 0].copy()

        self._audio_queue.put(audio)

    # -----------------------------------------------------
    # Frame extraction
    # -----------------------------------------------------

    def _next_frame(self) -> Optional[np.ndarray]:
        """
        Accumulate microphone samples until one VAD frame is ready.
        """
        while len(self._frame_buffer) < FRAME_LENGTH_SAMPLE:
            try:
                chunk = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                return None

            self._frame_buffer = np.concatenate([self._frame_buffer, chunk])

        frame = self._frame_buffer[:FRAME_LENGTH_SAMPLE]

        self._frame_buffer = self._frame_buffer[FRAME_SHIFT_SAMPLE:]

        return frame.astype(np.float32)

    # -----------------------------------------------------
    # Generator
    # -----------------------------------------------------

    def run_streaming_audio(self) -> Generator[SpeechSegment, None, None]:
        """
        Generator that yields speech segments as they complete.

        Yields
        ------
        SpeechSegment
        """

        self.vad.reset()
        self._running = True

        with sd.InputStream(
            samplerate=self.config.samplerate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            device=self.config.device,
            callback=self._audio_callback,
            blocksize=self.config.blocksize,
        ):
            while self._running:
                frame = self._next_frame()

                if frame is None:
                    logger.debug("No frame received from _next_frame.")
                    continue

                try:
                    result = self.vad.detect_frame(frame)
                except IndexError:
                    logger.debug(
                        "VAD results exhausted. Stopping stream at frame_index=%s",
                        self._frame_index,
                    )
                    self.stop()
                    return

                frame_time = self._frame_index * 0.01
                self._frame_index += 1

                # logger.debug(
                #     "Frame processed idx=%s start=%s end=%s",
                #     self._frame_index,
                #     result.is_speech_start,
                #     result.is_speech_end,
                # )

                # --------------------------------------------------
                # speech start
                # --------------------------------------------------

                if result.is_speech_start:
                    logger.debug("Speech start detected at %.3f", frame_time)
                    self._current_segment_audio = []
                    self._current_start_time = frame_time

                # --------------------------------------------------
                # accumulate speech
                # --------------------------------------------------

                if self._current_start_time is not None:
                    self._current_segment_audio.append(frame)

                # --------------------------------------------------
                # speech end
                # --------------------------------------------------

                if result.is_speech_end and self._current_start_time is not None:
                    audio = np.concatenate(self._current_segment_audio)

                    logger.debug(
                        "Speech segment finalized start=%.3f end=%.3f samples=%s",
                        self._current_start_time,
                        frame_time,
                        len(audio),
                    )

                    segment: SpeechSegment = {
                        "start_time": self._current_start_time,
                        "end_time": frame_time,
                        "audio": audio,
                    }

                    yield segment

                    self._current_segment_audio = []
                    self._current_start_time = None

    # -----------------------------------------------------

    def stop(self):
        self._running = False
