from collections import deque
from datetime import datetime

import numpy as np
from fireredvad.core.stream_vad_postprocessor import StreamVadFrameResult
from jet.audio.audio_waveform.hybrid_stream_vad_postprocessor import (
    HybridStreamVadPostprocessor,
)
from jet.audio.audio_waveform.speech_events import (
    SpeechSegmentEndEvent,
    SpeechSegmentStartEvent,
)
from jet.audio.audio_waveform.speech_handlers.base import SpeechSegmentHandler
from jet.audio.audio_waveform.speech_types import SpeechFrame, VadStateLabel
from rich.console import Console

console = Console()


class SpeechSegmentTracker:
    """Collects audio & probabilities, detects segment boundaries, notifies handlers."""

    def __init__(self):
        self.sample_rate = 16000
        self.is_speaking = False
        self.segment_counter = 0
        self.current_audio_chunks: list[np.ndarray] = []
        self.current_frames: list[SpeechFrame] = []
        self.current_start_frame = -1
        self.current_start_event: SpeechSegmentStartEvent | None = None
        self.current_forced_split = False
        self.current_trigger_reason = "silence"
        self.current_vad_states = []
        self.postprocessor: HybridStreamVadPostprocessor | None = None

        self.handlers: list[SpeechSegmentHandler] = []

        # NEW: pre-speech buffer for rise detection (≈6 s lookback)
        self.pre_audio_buffer: deque[np.ndarray] = deque(maxlen=200)
        self.last_segment_forced_split: bool = False

    def add_handler(self, handler: SpeechSegmentHandler) -> None:
        self.handlers.append(handler)

    def add_audio(self, samples: np.ndarray) -> None:
        """Always store wave amplitude; store audio only in the appropriate buffer."""
        if len(samples) == 0:
            return
        if not self.is_speaking:
            self.pre_audio_buffer.append(samples.astype(np.float32).copy())
        else:
            self.current_audio_chunks.append(samples.astype(np.float32))

    def on_frame(self, result: StreamVadFrameResult) -> None:
        """Called for every VAD frame result."""

        if result.is_speech_start:
            self._start_new_segment(result)

        if self.is_speaking:
            entry: SpeechFrame = {
                "frame_idx": result.frame_idx,
                "raw_prob": result.raw_prob,
                "smoothed_prob": result.smoothed_prob,
                "is_speech": result.is_speech,
                "is_speech_start": result.is_speech_start,
                "is_speech_end": result.is_speech_end,
                "vad_state": self._get_vad_state_name(),
            }
            self.current_frames.append(entry)

        if result.is_speech_end:
            self._end_segment(result)

    def _get_vad_state_name(self) -> VadStateLabel:
        if self.postprocessor is not None and hasattr(self.postprocessor, "state"):
            return getattr(
                self.postprocessor.state, "name", str(self.postprocessor.state)
            )
        return "UNKNOWN"

    def _start_new_segment(self, result: StreamVadFrameResult) -> None:
        if self.is_speaking:
            return

        self.is_speaking = True
        self.segment_counter += 1
        now_str = datetime.now().isoformat()

        start_event = SpeechSegmentStartEvent(
            segment_id=self.segment_counter,
            start_frame=int(result.frame_idx),
            start_time_sec=(result.frame_idx - 1) / 100.0,
            started_at=now_str,
            segment_dir=None,
        )

        for handler in self.handlers:
            handler.on_segment_start(start_event)

        self.current_audio_chunks = []
        self.current_frames = []
        self.current_start_frame = result.speech_start_frame
        self.current_start_event = start_event
        self.current_vad_states = []

        # === NEW: snap normal segments to waveform rise ===
        is_after_force_split = self.last_segment_forced_split
        if not is_after_force_split:
            self._prepend_rise_audio_from_pre()

        console.print(
            f"[TRACKER] [bold green]START[/] segment [cyan]{self.segment_counter}[/] "
            f"(dir: [magenta]{self.current_start_event.segment_dir or '—'}[/])",
            style="bold",
        )

    def _prepend_rise_audio_from_pre(self) -> None:
        """Prepend only the blocks that belong to the current waveform rise.
        This guarantees the saved segment audio literally starts at the rise
        for segments that follow a natural silence (not a force-split)."""
        if len(self.pre_audio_buffer) == 0:
            return

        WAVE_RISE_THRES = 0.01  # matches THRES_WAVE_MEDIUM in the UI app

        # Find the last low-amplitude block (end of previous silence)
        last_low_idx = -1
        for i, chunk in enumerate(self.pre_audio_buffer):
            if np.max(np.abs(chunk)) < WAVE_RISE_THRES:
                last_low_idx = i

        if last_low_idx == -1:
            return  # no silence detected (rare)

        rise_blocks = list(self.pre_audio_buffer)[last_low_idx + 1 :]
        if rise_blocks:
            self.current_audio_chunks = list(rise_blocks)
            console.print(
                f"[TRACKER] [yellow]Prepended {len(rise_blocks)} blocks "
                f"to align start with waveform rise[/]",
                style="yellow",
            )

    def _end_segment(self, result: StreamVadFrameResult) -> None:
        if not self.is_speaking:
            return

        self.is_speaking = False

        # Record whether THIS segment was force-split (for the NEXT start)
        self.last_segment_forced_split = self.current_forced_split

        if self.current_audio_chunks:
            audio = np.concatenate(self.current_audio_chunks)
        else:
            audio = np.empty(0, dtype=np.float32)

        actual_samples = len(audio)
        actual_duration = (
            actual_samples / self.sample_rate if actual_samples > 0 else 0.0
        )
        end_time_sec = self.current_start_event.start_time_sec + actual_duration

        end_frame = self.current_frames[-1]["frame_idx"] if self.current_frames else 0

        if self.postprocessor is not None:
            self.current_forced_split = getattr(
                self.postprocessor, "was_force_splitted", False
            )
            self.current_trigger_reason = getattr(
                self.postprocessor, "last_split_reason", "silence"
            )

        end_event = SpeechSegmentEndEvent(
            segment_id=self.segment_counter,
            start_frame=self.current_start_event.start_frame,
            end_frame=int(end_frame),
            start_time_sec=self.current_start_event.start_time_sec,
            end_time_sec=end_time_sec,
            duration_sec=actual_duration,
            audio=audio,
            prob_frames=self.current_frames[:],
            forced_split=self.current_forced_split,
            trigger_reason=self.current_trigger_reason,
            started_at=self.current_start_event.started_at,
            segment_dir=self.current_start_event.segment_dir,
        )

        for handler in self.handlers:
            handler.on_segment_end(end_event)

        reason = self.current_trigger_reason

        if self.current_forced_split:
            reason += " [forced]"

        console.print(
            f"[TRACKER] [bold red]END[/] segment [cyan]{self.segment_counter}[/] "
            f"duration [yellow]{end_event.duration_sec:.2f}s[/] • [italic magenta]{reason}[/]\n",
            style="bold",
        )

        self.reset()

    def reset(self):
        self.current_forced_split = False
        self.current_trigger_reason = "silence"
        self.current_start_event = None
        self.current_audio_chunks = []
        self.current_frames = []
        self.current_start_frame = -1
        self.current_vad_states = []
        self.pre_audio_buffer.clear()  # NEW
