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


def calculate_amplitude(samples: np.ndarray) -> float:
    """Peak amplitude (max |x|). Useful for clipping detection or transients."""
    if len(samples) == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def calculate_rms(samples: np.ndarray) -> float:
    """Root Mean Square – best simple measure of perceived loudness/energy.

    Range: 0.0 (true silence) → ~0.707 (full-scale sine wave)
    Typical speech values:
      - < 0.005     → silence / noise floor
      - 0.005–0.03  → very quiet / breath
      - 0.03–0.15   → normal conversational speech
      - 0.15–0.4+   → loud speech / shouting
    """
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def get_loudness_label(rms_value: float) -> str:
    """Return a human-readable loudness label based on RMS."""
    if rms_value < 0.005:
        return "silent"
    elif rms_value < 0.03:
        return "very_quiet"
    elif rms_value < 0.12:
        return "normal"
    elif rms_value < 0.25:
        return "loud"
    else:
        return "very_loud"


def has_sound(samples: np.ndarray, threshold: float = 0.005) -> bool:
    """Return True if the audio contains meaningful sound.

    Now aligned with get_loudness_label():
      - rms < 0.005  → "silent"       → has_sound=False
      - rms >= 0.005 → "very_quiet" and above → has_sound=True
    """
    if len(samples) == 0:
        return False
    rms_value = calculate_rms(samples)
    return rms_value >= threshold  # Note: >= so exactly 0.005 counts as sound


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

        # Pre-speech buffers for hybrid rise detection
        self.pre_audio_buffer: deque[np.ndarray] = deque(maxlen=200)
        self.pre_prob_buffer: deque[float] = deque(maxlen=200)
        self.last_segment_forced_split: bool = False

    def add_handler(self, handler: SpeechSegmentHandler) -> None:
        self.handlers.append(handler)

    def add_audio(self, samples: np.ndarray) -> None:
        if len(samples) == 0:
            return
        if not self.is_speaking:
            self.pre_audio_buffer.append(samples.astype(np.float32).copy())
        else:
            self.current_audio_chunks.append(samples.astype(np.float32))

    def add_prob(self, smoothed_prob: float) -> None:
        if not self.is_speaking:
            self.pre_prob_buffer.append(float(smoothed_prob))

    def on_frame(self, result: StreamVadFrameResult) -> None:
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

        if not self.last_segment_forced_split:
            self._prepend_hybrid_rise()

        console.print(
            f"[TRACKER] [bold green]START[/] segment [cyan]{self.segment_counter}[/] "
            f"(dir: [magenta]{self.current_start_event.segment_dir or '—'}[/])",
            style="bold",
        )

    def _prepend_hybrid_rise(self) -> None:
        """Hybrid rise detection using RMS for more stable loudness check."""
        if len(self.pre_audio_buffer) == 0 or len(self.pre_prob_buffer) == 0:
            return

        PROB_THRES = 0.3
        RMS_THRES = 0.01

        last_low_idx = -1
        for i in range(len(self.pre_prob_buffer)):
            prob = self.pre_prob_buffer[i]
            rms = calculate_rms(self.pre_audio_buffer[i])
            if prob < PROB_THRES and rms < RMS_THRES:
                last_low_idx = i

        if last_low_idx == -1:
            return

        rise_audio = list(self.pre_audio_buffer)[last_low_idx:]
        if rise_audio:
            self.current_audio_chunks = rise_audio
            last_rms = calculate_rms(self.pre_audio_buffer[last_low_idx])
            console.print(
                f"[TRACKER] [yellow]Hybrid prepend: {len(rise_audio)} blocks "
                f"(last prob={self.pre_prob_buffer[last_low_idx]:.3f}, "
                f"rms={last_rms:.4f} → {get_loudness_label(last_rms)})[/]",
                style="yellow",
            )

    def _end_segment(self, result: StreamVadFrameResult) -> None:
        if not self.is_speaking:
            return

        self.is_speaking = False

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

        self.last_segment_forced_split = self.current_forced_split

        # Compute loudness metrics for the whole segment
        segment_rms = calculate_rms(audio)
        loudness_label = get_loudness_label(segment_rms)
        segment_has_sound = has_sound(audio)  # now uses threshold=0.005

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
            # NEW fields
            rms=segment_rms,
            loudness_label=loudness_label,
            has_sound=segment_has_sound,
        )

        for handler in self.handlers:
            handler.on_segment_end(end_event)

        reason = self.current_trigger_reason
        if self.current_forced_split:
            reason += " [forced]"

        console.print(
            f"[TRACKER] [bold red]END[/] segment [cyan]{self.segment_counter}[/] "
            f"duration [yellow]{end_event.duration_sec:.2f}s[/] • rms={segment_rms:.4f} "
            f"({loudness_label}) • has_sound={segment_has_sound} • [italic magenta]{reason}[/]\n",
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
        self.pre_audio_buffer.clear()
        self.pre_prob_buffer.clear()
