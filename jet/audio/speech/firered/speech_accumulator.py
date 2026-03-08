import time
from collections import deque
from typing import Optional, TypedDict


class SegmentStats(TypedDict):
    speech_chunk_count: int
    vad_sum: float
    vad_min: float
    vad_max: float
    energy_sum: float
    energy_sum_squares: float
    energy_min: float
    energy_max: float
    duration_ms: float
    duration_sec: float
    has_data: bool


class LiveSpeechSegmentAccumulator:
    """Manages audio buffer + stats for one active speech segment."""

    BYTES_PER_SAMPLE: int = 2
    SAMPLES_PER_CHUNK: int = 512
    SMART_TRIM_VAD_THRESHOLD: float = 0.15

    def __init__(
        self,
        sample_rate: int,
        pre_roll_buffer: Optional[deque[bytes]] = None,
        max_pre_roll_duration_sec: Optional[float] = None,
        start_time: float | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.buffer = bytearray()
        self._chunk_bytes = self.SAMPLES_PER_CHUNK * self.BYTES_PER_SAMPLE
        self.speech_chunk_count = 0
        self.vad_sum = 0.0
        self.vad_min = 1.0
        self.vad_max = 0.0
        self.energy_sum = 0.0
        self.energy_sum_squares = 0.0
        self.energy_min = float("inf")
        self.energy_max = 0.0
        self._vad_probs: list[float] = []
        self._rms_values: list[float] = []
        self.start_time = start_time if start_time is not None else time.monotonic()
        self.end_time = self.start_time
        if pre_roll_buffer is not None:
            selected_chunks = self._select_pre_roll_chunks(
                pre_roll_buffer=pre_roll_buffer,
                max_pre_roll_duration_sec=max_pre_roll_duration_sec,
            )
            for chunk in selected_chunks:
                self.buffer.extend(chunk)
                self._vad_probs.append(0.0)
                self._rms_values.append(0.0)
        self._update_end_time()

    def _select_pre_roll_chunks(
        self,
        pre_roll_buffer: deque[bytes],
        max_pre_roll_duration_sec: Optional[float],
    ) -> list[bytes]:
        if not pre_roll_buffer:
            return []
        all_chunks = list(pre_roll_buffer)
        if max_pre_roll_duration_sec is None:
            chunks_allowed = len(all_chunks)
        elif max_pre_roll_duration_sec <= 0:
            return []
        else:
            samples_allowed = int(max_pre_roll_duration_sec * self.sample_rate)
            chunks_allowed = samples_allowed // self.SAMPLES_PER_CHUNK
        if chunks_allowed <= 0:
            return []
        trailing = all_chunks[-chunks_allowed:]
        valid_chunks: list[bytes] = []
        for i, chunk in enumerate(trailing):
            if len(chunk) != self._chunk_bytes:
                print(
                    f"Warning: pre-roll chunk {i} has unexpected size "
                    f"{len(chunk)} bytes (expected {self._chunk_bytes})"
                )
                continue
            valid_chunks.append(chunk)
        return valid_chunks

    def _update_end_time(self) -> None:
        self.end_time = self.start_time + self.get_duration_sec()

    def get_duration_sec(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return len(self.buffer) / (self.sample_rate * self.BYTES_PER_SAMPLE)

    def duration_samples(self) -> int:
        return len(self.buffer) // self.BYTES_PER_SAMPLE

    def append(self, chunk: bytes, speech_prob: float, rms: float) -> None:
        self.buffer.extend(chunk)
        self.speech_chunk_count += 1
        self.vad_sum += speech_prob
        self.vad_min = min(self.vad_min, speech_prob)
        self.vad_max = max(self.vad_max, speech_prob)
        self.energy_sum += rms
        self.energy_sum_squares += rms**2
        self.energy_min = min(self.energy_min, rms)
        self.energy_max = max(self.energy_max, rms)
        self._vad_probs.append(speech_prob)
        self._rms_values.append(rms)
        self._update_end_time()

    def get_stats(self) -> SegmentStats:
        count = self.speech_chunk_count
        return {
            "speech_chunk_count": count,
            "vad_sum": self.vad_sum,
            "vad_min": self.vad_min if count > 0 else 0.0,
            "vad_max": self.vad_max if count > 0 else 0.0,
            "energy_sum": float(self.energy_sum),
            "energy_sum_squares": float(self.energy_sum_squares),
            "energy_min": float(self.energy_min) if count > 0 else 0.0,
            "energy_max": float(self.energy_max) if count > 0 else 0.0,
            "duration_ms": self.get_duration_sec() * 1000.0,
            "duration_sec": self.get_duration_sec(),
            "has_data": count > 0,
        }

    def get_start_wallclock(self) -> float:
        return self.start_time

    def get_end_wallclock(self) -> float:
        return self.end_time

    def get_speech_probs(self) -> list[float]:
        """
        Returns the list of per-chunk speech probabilities (VAD scores)
        in the order they were appended (including any pre-roll chunks with 0.0).
        """
        return self._vad_probs.copy()

    def get_rms_values(self) -> list[float]:
        """
        Returns the list of per-chunk RMS values
        in the order they were appended (including any pre-roll chunks with 0.0).
        """
        return self._rms_values.copy()

    def trim_audio(self, max_duration: float) -> None:
        if max_duration <= 0:
            self.reset()
            return

        current_sec = self.get_duration_sec()
        if current_sec <= max_duration:
            return

        # --- compute chunk duration ---
        chunk_duration_sec = self.SAMPLES_PER_CHUNK / self.sample_rate

        # --- determine max chunks allowed ---
        max_chunks = int(max_duration // chunk_duration_sec)

        if max_chunks <= 0:
            self.reset()
            return

        total_chunks = len(self._vad_probs)

        # nothing to trim
        if total_chunks <= max_chunks:
            return

        # --- base trim requirement ---
        trim_chunks = total_chunks - max_chunks

        # --- attempt smart trim at silence boundary ---
        silence_trim_index: Optional[int] = None

        for i in range(trim_chunks - 1, -1, -1):
            if self._vad_probs[i] < self.SMART_TRIM_VAD_THRESHOLD:
                silence_trim_index = i + 1
                break

        if silence_trim_index is not None:
            trim_chunks = silence_trim_index

        trim_samples = trim_chunks * self.SAMPLES_PER_CHUNK
        trim_bytes = trim_samples * self.BYTES_PER_SAMPLE

        # --- adjust start time before modifying buffer ---
        trimmed_duration = trim_samples / self.sample_rate
        self.start_time += trimmed_duration

        # --- remove audio from the front (avoid reallocating buffer) ---
        del self.buffer[:trim_bytes]

        # --- trim stat arrays ---
        self._vad_probs = self._vad_probs[trim_chunks:]
        self._rms_values = self._rms_values[trim_chunks:]

        # --- recompute aggregates ---
        self._recompute_aggregates()

        # --- update end time ---
        self._update_end_time()

    def _recompute_aggregates(self) -> None:
        self.speech_chunk_count = len(self._vad_probs)
        if self.speech_chunk_count > 0:
            self.vad_sum = sum(self._vad_probs)
            self.vad_min = min(self._vad_probs)
            self.vad_max = max(self._vad_probs)
            self.energy_sum = sum(self._rms_values)
            self.energy_sum_squares = sum(r**2 for r in self._rms_values)
            self.energy_min = min(self._rms_values)
            self.energy_max = max(self._rms_values)
        else:
            self.vad_sum = 0.0
            self.vad_min = 1.0
            self.vad_max = 0.0
            self.energy_sum = 0.0
            self.energy_sum_squares = 0.0
            self.energy_min = float("inf")
            self.energy_max = 0.0

    def reset(self) -> None:
        self.buffer.clear()
        self._vad_probs.clear()
        self._rms_values.clear()
        self.speech_chunk_count = 0
        self.vad_sum = 0.0
        self.vad_min = 1.0
        self.vad_max = 0.0
        self.energy_sum = 0.0
        self.energy_sum_squares = 0.0
        self.energy_min = float("inf")
        self.energy_max = 0.0
        self.start_time = time.monotonic()
        self._update_end_time()
