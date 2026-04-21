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

    def _has_ongoing_sound_at_end(self) -> bool:
        """Return True if the trailing audio still contains normal-to-high energy
        (i.e. the low-prob 'silence' or 'valley' is likely a VAD glitch)."""
        if len(self.current_audio_chunks) < 5:
            return False  # too short to judge reliably

        # Last ~0.48 s (15 blocks @ 512 samples / 16 kHz) covers the
        # consecutive low-prob region that triggered the end.
        recent_chunks = list(self.current_audio_chunks)[-15:]
        if not recent_chunks:
            return False

        recent_audio = np.concatenate(recent_chunks)
        if len(recent_audio) < 1000:  # ~62 ms minimum
            return False

        recent_rms = compute_rms(recent_audio)

        # Normal-to-high RMS while probs are low = false silence/valley.
        return recent_rms >= 0.12

    def _is_premature_end(self) -> bool:
        """True if we should ignore the current is_speech_end because energy
        is still high (prevents cutting real speech)."""
        if self.postprocessor is None:
            return False

        trigger = getattr(
            self.postprocessor,
            "last_split_reason",
            getattr(self.postprocessor, "last_force_split_reason", "silence"),
        )
        if trigger not in ("silence"):
            return False  # valley_detection, hard_limit etc. are always honored

        return self._has_ongoing_sound_at_end()

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
            if self._is_premature_end():
                trigger = getattr(
                    self.postprocessor,
                    "last_split_reason",
                    getattr(self.postprocessor, "last_force_split_reason", "silence"),
                )
                console.print(
                    f"[TRACKER] [yellow]IGNORING premature end[/] (trigger={trigger}, "
                    f"high RMS at end despite low probs) – continuing segment",
                    style="yellow",
                )
            else:
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
        """Energy- & probability-based rise detection.
        Finds the last silent block and/or last low-probability block in the pre-buffer,
        then prepends only the rising edge blocks (after both) to both audio and frame history.
        This fixes situations where audio/prob blip up before VAD triggers speech,
        preventing late segment starts and including early rise in frame/alignment tracking.
        """
        if len(self.pre_audio_buffer) == 0 or len(self.pre_prob_buffer) == 0:
            return

        audio_list = list(self.pre_audio_buffer)
        prob_list = list(self.pre_prob_buffer)

        # --- STEP 1: Find last silent audio block ---
        last_silent_audio_idx = -1
        for i in range(len(audio_list)):
            if not has_sound(audio_list[i]):
                last_silent_audio_idx = i

        # --- STEP 2: Find last low-prob frame (rising edge start) ---
        last_low_prob_idx = -1
        for i in range(len(prob_list)):
            if prob_list[i] < self.pre_prob_thres:
                last_low_prob_idx = i

        # --- STEP 3: Combine both signals (take later start to avoid noise) ---
        start_idx_candidates = []
        if last_silent_audio_idx != -1:
            start_idx_candidates.append(last_silent_audio_idx + 1)
        if last_low_prob_idx != -1:
            start_idx_candidates.append(last_low_prob_idx + 1)

        if not start_idx_candidates:
            return

        rise_start_idx = max(start_idx_candidates)

        if rise_start_idx >= len(audio_list):
            return

        # --- STEP 4: Extract rising audio ---
        rise_audio = audio_list[rise_start_idx:]

        # --- STEP 5: Build pseudo frames for early probabilities ---
        rise_probs = prob_list[rise_start_idx:]
        rise_frames: list[SpeechFrame] = []

        base_frame_idx = self.current_start_frame - len(rise_probs)
        if base_frame_idx < 0:
            base_frame_idx = 0

        for i, prob in enumerate(rise_probs):
            rise_frames.append(
                {
                    "frame_idx": base_frame_idx + i,
                    "raw_prob": prob,
                    "smoothed_prob": prob,
                    "is_speech": prob >= self.pre_prob_thres,
                    "is_speech_start": False,
                    "is_speech_end": False,
                    "vad_state": "SPEECH",
                }
            )

        # --- STEP 6: prepend both audio + frames ---
        self.current_audio_chunks = rise_audio
        self.current_frames = rise_frames

        debug_rms = (
            compute_rms(audio_list[rise_start_idx - 1]) if rise_start_idx > 0 else 0.0
        )

        console.print(
            f"[TRACKER] [yellow]Hybrid prepend (audio+prob): {len(rise_audio)} blocks "
            f"(start_idx={rise_start_idx}, pre_frames={len(rise_frames)}, "
            f"rms_before={debug_rms:.4f})[/]",
            style="yellow",
        )

    def _end_segment(self, result: StreamVadFrameResult) -> None:
        if not self.is_speaking:
            return

        # === RESPECT DEEPEST VALLEY ===
        # The postprocessor now sets speech_end_frame to the cleanest
        # valley point when valley_detection triggers. We must trim
        # the audio and frames to that exact frame.
        true_end_frame = getattr(result, "speech_end_frame", None)
        if true_end_frame is None or true_end_frame <= 0:
            true_end_frame = (
                self.current_frames[-1]["frame_idx"] if self.current_frames else 0
            )

        self.is_speaking = False

        # Trim both lists to the true end frame (no-op for normal silence ends)
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
                    f"[TRACKER] [dim cyan]Trimmed {trimmed} frame(s) "
                    f"after deepest valley (ended at frame {true_end_frame})[/]",
                    style="dim",
                )
            self.current_frames = self.current_frames[:cut_idx]
            self.current_audio_chunks = self.current_audio_chunks[:cut_idx]

        if self.current_audio_chunks:
            audio = np.concatenate(self.current_audio_chunks)
        else:
            audio = np.empty(0, dtype=np.float32)

        end_frame = true_end_frame
        actual_samples = len(audio)
        actual_duration = (
            actual_samples / self.sample_rate if actual_samples > 0 else 0.0
        )
        end_time_sec = self.current_start_event.start_time_sec + actual_duration

        # === NEW FILTER: prevent sending short / silent segments ===
        segment_rms = compute_rms(audio)
        loudness_label = rms_to_loudness_label(segment_rms)
        segment_has_sound = has_sound(audio)

        if not segment_has_sound or actual_duration < 0.25:
            console.print(
                f"[TRACKER] [dim yellow]SKIPPED silent/short segment[/] {self.segment_counter} "
                f"dur={actual_duration:.2f}s rms={segment_rms:.4f} has_sound={segment_has_sound} "
                f"reason={self.current_trigger_reason}",
                style="dim",
            )
            self.reset()
            return
        # ========================================================

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
