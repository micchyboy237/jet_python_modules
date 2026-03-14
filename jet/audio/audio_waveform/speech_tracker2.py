# jet_python_modules/jet/audio/audio_waveform/speech_tracker2.py
from datetime import datetime
from pathlib import Path

import numpy as np
from fireredvad.core.stream_vad_postprocessor import StreamVadFrameResult
from jet.audio.audio_waveform.speech_events import (
    SpeechSegmentEndEvent,
    SpeechSegmentStartEvent,
)
from jet.audio.audio_waveform.speech_handlers.base import SpeechSegmentHandler


class SpeechSegmentTracker:
    """Collects audio & probabilities, detects segment boundaries, notifies handlers."""

    def __init__(self):
        self.sample_rate = 16000
        self.is_speaking = False
        self.segment_counter = 0
        self.current_audio: np.ndarray = np.empty(0, dtype=np.float32)
        self.current_probs: list[dict] = []
        self.current_segment_dir: Path | None = None
        self.current_start_frame = -1
        self.current_summary: dict = {}
        self.current_forced_split = False
        self.current_trigger_reason = "silence"
        self.current_vad_states = []
        self.postprocessor = None

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
        if result.is_speech_start:
            self._start_new_segment(result)
        if result.is_speech_end:
            self._end_segment(result)

        if self.is_speaking:
            entry = {
                "frame_idx": result.frame_idx,
                "raw_prob": result.raw_prob,
                "smoothed_prob": result.smoothed_prob,
                "is_speech": result.is_speech,
                "is_speech_start": result.is_speech_start,
                "is_speech_end": result.is_speech_end,
                "speech_start_frame": result.speech_start_frame,
                "speech_end_frame": result.speech_end_frame,
                "vad_state": self._get_vad_state_name(),
            }
            self.current_probs.append(entry)

    def _get_vad_state_name(self) -> str:
        if self.postprocessor is not None and hasattr(self.postprocessor, "state"):
            return getattr(
                self.postprocessor.state, "name", str(self.postprocessor.state)
            )
        return "?"

    def _start_new_segment(self, result: StreamVadFrameResult) -> None:
        if self.is_speaking:
            return

        self.is_speaking = True
        self.segment_counter += 1
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        start_event = SpeechSegmentStartEvent(
            segment_id=self.segment_counter,
            start_frame=int(result.speech_start_frame),
            start_time_sec=round((result.speech_start_frame - 1) / 100.0, 3),
            datetime_started=now_str,
            segment_dir=None,
        )

        # Let handlers initialize (e.g. create directory)
        for handler in self.handlers:
            handler.on_segment_start(start_event)

        # Remember the directory if any handler set it
        self.current_segment_dir = start_event.segment_dir

        self.current_audio = np.empty(0, dtype=np.float32)
        self.current_probs = []
        self.current_start_frame = result.speech_start_frame
        self.current_summary = {
            "segment_id": self.segment_counter,
            "start_frame": int(result.speech_start_frame),
            "start_time_sec": start_event.start_time_sec,
            "datetime_started": now_str,
        }
        self.current_vad_states = []

        print(
            f"[TRACKER] START → segment {self.segment_counter} (dir: {self.current_segment_dir})"
        )

    def _end_segment(self, result: StreamVadFrameResult) -> None:
        if not self.is_speaking:
            return

        self.is_speaking = False
        end_frame = result.speech_end_frame
        end_time_sec = round((end_frame - 1) / 100.0, 3)

        if self.postprocessor is not None:
            self.current_forced_split = getattr(
                self.postprocessor, "was_force_splitted", False
            )
            self.current_trigger_reason = getattr(
                self.postprocessor, "last_split_reason", "silence"
            )

        # Compute statistics
        if self.current_probs:
            probs = [p["smoothed_prob"] for p in self.current_probs]
            is_speech_list = [p["is_speech"] for p in self.current_probs]
            avg_prob = round(sum(probs) / len(probs), 3)
            max_prob = round(max(probs), 3)
            min_prob = round(min(probs), 3)
            speech_ratio = round(sum(is_speech_list) / len(is_speech_list), 3)
        else:
            avg_prob = max_prob = min_prob = speech_ratio = 0.0

        if len(self.current_audio) > 0:
            rms = np.sqrt(np.mean(self.current_audio**2))
            energy_db = round(20 * np.log10(rms + 1e-10), 2)
        else:
            energy_db = -float("inf")

        self.current_summary.update(
            {
                "end_frame": int(end_frame),
                "end_time_sec": end_time_sec,
                "duration_sec": round(
                    end_time_sec - self.current_summary["start_time_sec"], 3
                ),
                "audio_samples": len(self.current_audio),
                "prob_frames": len(self.current_probs),
                "forced_split": self.current_forced_split,
                "trigger_reason": self.current_trigger_reason,
                "avg_smoothed_prob": float(avg_prob),
                "max_smoothed_prob": float(max_prob),
                "min_smoothed_prob": float(min_prob),
                "speech_frame_ratio": speech_ratio,
                "energy_db_avg": energy_db,
            }
        )

        end_event = SpeechSegmentEndEvent(
            segment_id=self.segment_counter,
            start_frame=self.current_summary["start_frame"],
            end_frame=int(end_frame),
            start_time_sec=self.current_summary["start_time_sec"],
            end_time_sec=end_time_sec,
            duration_sec=self.current_summary["duration_sec"],
            audio=self.current_audio.copy(),
            probs=self.current_probs[:],
            forced_split=self.current_forced_split,
            trigger_reason=self.current_trigger_reason,
            summary=self.current_summary.copy(),
            segment_dir=self.current_segment_dir,
        )

        # Notify all handlers
        for handler in self.handlers:
            handler.on_segment_end(end_event)

        print(f"[TRACKER] END → segment {self.segment_counter}\n")

        # Reset
        self.current_forced_split = False
        self.current_trigger_reason = "silence"
        self.current_segment_dir = None
        self.current_audio = np.empty(0, dtype=np.float32)
        self.current_probs = []
        self.current_start_frame = -1
        self.current_vad_states = []
