# jet_python_modules/jet/audio/speech/speechbrain/speech_segmentor.py

import time
from collections import deque
from typing import Callable, Optional

from jet.audio.speech.speechbrain.speech_accumulator import LiveSpeechSegmentAccumulator

SendCallback = Callable[
    [
        bytes,  # audio bytes to send
        bool,  # is_final
        float,  # start_time
        float,  # duration_sec
        int,  # chunk_index
        dict,  # stats
    ],
    None,
]


class LiveSpeechSegmentor:
    """Decides when to start, accumulate, send partial/final segments or discard"""

    def __init__(
        self,
        config,
        sample_rate: int,
        on_send: SendCallback,
        pre_roll_buffer_maxlen: int = 20,
    ):
        self.config = config
        self.sample_rate = sample_rate
        self.on_send = on_send

        self.current: Optional[LiveSpeechSegmentAccumulator] = None
        self.pre_roll: deque[bytes] = deque(maxlen=pre_roll_buffer_maxlen)

        self.chunk_index: int = 0
        self.silence_start: Optional[float] = None
        self.last_send_time: Optional[float] = None

    def add_chunk(
        self,
        chunk: bytes,
        is_speech: bool,
        speech_prob: float,
        rms: float,
        wallclock: float | None = None,
    ) -> None:
        now = wallclock or time.monotonic()

        # Always keep recent silent chunks for pre-roll
        self.pre_roll.append(chunk)

        if self.current is None:
            if not is_speech:
                return
            self.current = LiveSpeechSegmentAccumulator(
                sample_rate=self.sample_rate,
                pre_roll_buffer=self.pre_roll,
                start_time=now,
            )
            self.chunk_index = 0
            self.last_send_time = now  # ← important: initialize here
            self.silence_start = None

        self.current.append(chunk, speech_prob, rms)
        duration_sec = self.current.get_duration_sec()

        # ── 1. Check for finalization FIRST (silence long enough)
        if not is_speech:
            if self.silence_start is None:
                self.silence_start = now
        else:
            self.silence_start = None

        finalized = False
        if (
            self.silence_start is not None
            and now - self.silence_start >= self.config.min_silence_duration_sec
        ):
            if duration_sec >= self.config.min_speech_duration_sec:
                self._send_current(is_final=True, now=now)
                self.reset()
                finalized = True
            else:
                self.reset()
                finalized = True

        if finalized:
            return  # ← prevent further partial checks after final send

        # ── 2. Max duration trim (still allowed during speech)
        if duration_sec > self.config.max_speech_duration_sec:
            self.current.trim_to_overlap(
                int(self.config.max_speech_duration_sec * self.sample_rate)
            )
            self._send_current(is_final=False, now=now)
            self.last_send_time = now
            return  # optional: can continue or return

        # ── 3. Time-based partial — only during active speech
        if is_speech and (
            self.last_send_time is None
            or now - self.last_send_time >= self.config.chunk_duration_sec
        ):
            if duration_sec >= self.config.min_speech_duration_sec:
                self._send_current(is_final=False, now=now)
                self.last_send_time = now

    def _send_current(self, is_final: bool, now: float) -> None:
        if self.current is None or len(self.current.buffer) == 0:
            return

        stats = self.current.get_stats()
        self.on_send(
            bytes(self.current.buffer),
            is_final,
            self.current.start_time,
            stats["duration_sec"],
            self.chunk_index,
            stats,
        )
        self.chunk_index += 1

        if is_final:
            self.reset()

    def reset(self) -> None:
        self.current = None
        self.chunk_index = 0
        self.last_send_time = None
        self.silence_start = None
