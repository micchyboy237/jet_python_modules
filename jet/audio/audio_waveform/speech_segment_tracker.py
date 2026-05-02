from datetime import datetime

import numpy as np
from fireredvad.core.stream_vad_postprocessor import StreamVadFrameResult
from jet.audio.audio_waveform.hybrid_stream_vad_postprocessor import (
    HybridStreamVadPostprocessor,
)
from jet.audio.audio_waveform.pre_roll_buffer import PreRollBuffer
from jet.audio.audio_waveform.speech_events import (
    SpeechSegmentEndEvent,
    SpeechSegmentStartEvent,
)
from jet.audio.audio_waveform.speech_handlers.base import SpeechSegmentHandler
from jet.audio.audio_waveform.speech_types import SpeechFrame, VadStateLabel
from jet.audio.helpers.energy import (
    compute_rms,
    has_sound,
    rms_to_loudness_label,
)
from jet.audio.helpers.energy_base import SILENCE_MAX_THRESHOLD
from rich.console import Console

console = Console()

PRE_RMS_THRES = SILENCE_MAX_THRESHOLD * 10


class SpeechSegmentTracker:
    """Collects audio & probabilities, detects segment boundaries, notifies handlers."""

    def __init__(self, speech_threshold: float = 0.5):
        self.pre_prob_thres = 0.1
        self.sample_rate = 16000
        self.active_vad: str = "fr"  # set externally by TrackerObserver
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

        # Pre-speech buffer for hybrid rise detection
        self.pre_roll_buffer = PreRollBuffer(maxlen=200)
        self.last_segment_forced_split: bool = False

    def add_handler(self, handler: SpeechSegmentHandler) -> None:
        self.handlers.append(handler)

    def add_audio(self, samples: np.ndarray) -> None:
        if len(samples) == 0:
            return

        if not self.is_speaking:
            self.pre_roll_buffer.add_audio(samples)
        else:
            self.current_audio_chunks.append(samples.astype(np.float32))

    def add_prob(self, smoothed_prob: float) -> None:
        if not self.is_speaking:
            self.pre_roll_buffer.add_prob(smoothed_prob)

    # Removed: _has_ongoing_sound_at_end() and _is_premature_end()

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
            self._end_segment(result)  # No more premature check here

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
            vad_type=self.active_vad,
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
        """Energy- & probability-based rise detection using the pre-roll buffer.

        Finds the last silent block and/or last low-probability block in the pre-buffer,
        then prepends only the rising edge blocks (after both) to both audio and frame history.

        This fixes situations where audio/probability blips up before the VAD officially
        triggers speech start — preventing late segment starts and including early rise.
        """
        if len(self.pre_roll_buffer.audio) == 0 or len(self.pre_roll_buffer.probs) == 0:
            return

        # Get the rising edge from pre-roll buffer
        rise_audio, rise_frames, rise_start_idx = self.pre_roll_buffer.get_rising_edge(
            pre_prob_thres=self.pre_prob_thres
        )

        if not rise_audio:
            console.print(
                "[TRACKER] [yellow]No rising edge found in pre-roll buffer[/]",
                style="yellow",
            )
            return

        # Apply the pre-roll data to current segment
        self.current_audio_chunks = rise_audio
        self.current_frames = rise_frames

        # Debug info
        debug_rms = 0.0
        if rise_start_idx > 0:
            audio_list = list(self.pre_roll_buffer.audio)
            debug_rms = compute_rms(audio_list[rise_start_idx - 1])

        console.print(
            f"[TRACKER] [yellow]Hybrid prepend (audio+prob): {len(rise_audio)} blocks "
            f"(start_idx={rise_start_idx}, pre_frames={len(rise_frames)}, "
            f"rms_before={debug_rms:.4f})[/]",
            style="yellow",
        )

    def _end_segment(self, result: StreamVadFrameResult) -> None:
        if not self.is_speaking:
            return

        # Simple end logic: always end at last frame available, ignore valleys
        true_end_frame = (
            self.current_frames[-1]["frame_idx"] if self.current_frames else 0
        )

        self.is_speaking = False

        # Cut frame/audio just in case (should already be aligned)
        if self.current_frames:
            cut_idx = 0
            for i, entry in enumerate(self.current_frames):
                if entry["frame_idx"] <= true_end_frame:
                    cut_idx = i + 1
                else:
                    break
            trimmed = len(self.current_frames) - cut_idx
            if trimmed > 0:
                console.print(
                    f"[TRACKER] [dim cyan]Trimmed {trimmed} frame(s) after end point[/]",
                    style="dim",
                )
            self.current_frames = self.current_frames[:cut_idx]
            self.current_audio_chunks = self.current_audio_chunks[:cut_idx]

        # Combine audio
        audio = (
            np.concatenate(self.current_audio_chunks)
            if self.current_audio_chunks
            else np.empty(0, dtype=np.float32)
        )

        # Final validation
        actual_duration = len(audio) / self.sample_rate
        segment_rms = compute_rms(audio)
        segment_has_sound = has_sound(audio)

        if not segment_has_sound or actual_duration < 0.25:
            console.print(
                "[TRACKER] [dim yellow]SKIPPED silent/short segment[/]", style="dim"
            )
            self.reset()
            return

        # Event creation and handler notification
        end_frame = true_end_frame
        actual_samples = len(audio)
        end_time_sec = self.current_start_event.start_time_sec + actual_duration
        loudness_label = rms_to_loudness_label(segment_rms)

        if self.postprocessor is not None:
            self.current_forced_split = getattr(
                self.postprocessor, "was_force_splitted", False
            )
            self.current_trigger_reason = getattr(
                self.postprocessor, "last_split_reason", "silence"
            )

        self.last_segment_forced_split = self.current_forced_split

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
            # Energy RMS fields
            segment_rms=segment_rms,
            loudness=loudness_label,
            has_sound=segment_has_sound,
            vad_type=self.current_start_event.vad_type,
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
        self.pre_roll_buffer.clear()
