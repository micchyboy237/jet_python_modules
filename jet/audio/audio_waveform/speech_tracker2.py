# jet_python_modules/jet/audio/audio_waveform/speech_tracker2.py
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
        self.current_audio: np.ndarray = np.empty(0, dtype=np.float32)
        self.current_frames: list[SpeechFrame] = []
        self.current_start_frame = -1
        self.current_start_event: SpeechSegmentStartEvent | None = None
        self.current_forced_split = False
        self.current_trigger_reason = "silence"
        self.current_vad_states = []
        self.postprocessor: HybridStreamVadPostprocessor | None = None

        self.handlers: list[SpeechSegmentHandler] = []

    def add_handler(self, handler: SpeechSegmentHandler) -> None:
        self.handlers.append(handler)

    def add_audio(self, samples: np.ndarray) -> None:
        """Append new microphone block — only while in speaking state."""
        if self.is_speaking and len(samples) > 0:
            self.current_audio = np.append(
                self.current_audio, samples.astype(np.float32)
            )

    def on_frame(self, result: StreamVadFrameResult) -> None:
        """Called for every VAD frame result."""

        # ✅ START must happen BEFORE append
        if result.is_speech_start:
            self._start_new_segment(result)

        # ✅ Append AFTER start so first frame is captured correctly
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

        # ✅ END after append so last frame is included
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
            # ✅ align time with actual stored frame
            start_frame=int(result.frame_idx),
            start_time_sec=(result.frame_idx - 1) / 100.0,
            started_at=now_str,
            segment_dir=None,
        )

        # Let handlers initialize (e.g. create directory)
        for handler in self.handlers:
            handler.on_segment_start(start_event)

        # Remember the directory if any handler set it

        self.current_audio = np.empty(0, dtype=np.float32)
        self.current_frames = []
        self.current_start_frame = result.speech_start_frame
        self.current_start_event = start_event
        self.current_vad_states = []

        console.print(
            f"[TRACKER] [bold green]START[/] segment [cyan]{self.segment_counter}[/]  "
            f"(dir: [magenta]{self.current_start_event.segment_dir or '—'}[/])",
            style="bold",
        )

    def _end_segment(self, result: StreamVadFrameResult) -> None:
        if not self.is_speaking:
            return

        self.is_speaking = False

        # === BEST FIX: use actual audio sample count (eliminates all duration mismatch) ===
        actual_samples = len(self.current_audio)
        actual_duration = (
            actual_samples / self.sample_rate if actual_samples > 0 else 0.0
        )
        end_time_sec = self.current_start_event.start_time_sec + actual_duration

        # Keep the original frame-based end_frame only for debug/prob list (unchanged)
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
            audio=self.current_audio.copy(),
            prob_frames=self.current_frames[:],
            forced_split=self.current_forced_split,
            trigger_reason=self.current_trigger_reason,
            started_at=self.current_start_event.started_at,
            segment_dir=self.current_start_event.segment_dir,
        )

        # Notify all handlers
        for handler in self.handlers:
            handler.on_segment_end(end_event)

        reason = self.current_trigger_reason

        if self.current_forced_split:
            reason += " [forced]"

        console.print(
            f"[TRACKER] [bold red]END[/]   segment [cyan]{self.segment_counter}[/]  "
            f"duration [yellow]{end_event.duration_sec:.2f}s[/]  • [italic magenta]{reason}[/]\n",
            style="bold",
        )

        self.reset()

    def reset(self):
        # Reset
        self.current_forced_split = False
        self.current_trigger_reason = "silence"
        self.current_start_event = None
        self.current_audio = np.empty(0, dtype=np.float32)
        self.current_frames = []
        self.current_start_frame = -1
        self.current_vad_states = []
