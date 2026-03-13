# jet_python_modules/jet/audio/audio_waveform/speech_tracker.py
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from fireredvad.core.stream_vad_postprocessor import StreamVadFrameResult


class SpeechSegmentTracker:
    """Listens to HybridStreamVadPostprocessor state transitions and saves complete segments."""

    def __init__(self, save_dir: str = "saved_speech_segments"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.sample_rate = 16000
        self.is_speaking = False
        self.segment_counter = 0

        self.current_audio: np.ndarray = np.empty(0, dtype=np.float32)
        self.current_probs: list[dict] = []
        self.current_segment_dir: Path | None = None
        self.current_start_frame = -1
        self.current_summary: dict = {}

    def add_audio(self, samples: np.ndarray) -> None:
        """Call this with every microphone block (after or before VAD). Only collects while speaking."""
        if self.is_speaking and len(samples) > 0:
            self.current_audio = np.append(
                self.current_audio, samples.astype(np.float32)
            )

    def on_frame(self, result: StreamVadFrameResult) -> None:
        """This is the listener — called on every frame result from detect_chunk/detect_frame."""
        if result.is_speech_start:
            self._start_new_segment(result)

        if result.is_speech_end:
            self._end_segment(result)

        # Collect probabilities while we are in a speech segment (including the transition frames)
        if self.is_speaking:
            self.current_probs.append(
                {
                    "frame_idx": result.frame_idx,
                    "raw_prob": result.raw_prob,
                    "smoothed_prob": result.smoothed_prob,
                    "is_speech": result.is_speech,
                }
            )

    def _start_new_segment(self, result: StreamVadFrameResult) -> None:
        if self.is_speaking:
            return
        self.is_speaking = True
        self.segment_counter += 1

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_segment_dir = (
            self.save_dir / f"segment_{now_str}_{self.segment_counter:03d}"
        )
        self.current_segment_dir.mkdir(parents=True, exist_ok=True)

        self.current_audio = np.empty(0, dtype=np.float32)
        self.current_probs = []
        self.current_start_frame = result.speech_start_frame

        self.current_summary = {
            "segment_id": self.segment_counter,
            "start_frame": int(result.speech_start_frame),
            "start_time_sec": round((result.speech_start_frame - 1) / 100.0, 3),
            "datetime_started": now_str,
        }

        print(f"[SPEECH TRACKER] START → {self.current_segment_dir}")

    def _end_segment(self, result: StreamVadFrameResult) -> None:
        if not self.is_speaking or self.current_segment_dir is None:
            return

        self.is_speaking = False

        end_frame = result.speech_end_frame
        end_time_sec = round((end_frame - 1) / 100.0, 3)

        self.current_summary.update(
            {
                "end_frame": int(end_frame),
                "end_time_sec": end_time_sec,
                "duration_sec": round(
                    end_time_sec - self.current_summary["start_time_sec"], 3
                ),
                "audio_samples": len(self.current_audio),
                "prob_frames": len(self.current_probs),
            }
        )

        # === SAVE FILES ===
        wav_path = self.current_segment_dir / "sound.wav"
        if len(self.current_audio) > 0:
            sf.write(str(wav_path), self.current_audio, self.sample_rate)
            print(f"   Saved sound.wav ({len(self.current_audio):,} samples)")
        else:
            print("   WARNING: No audio collected for this segment")

        (self.current_segment_dir / "summary.json").write_text(
            json.dumps(self.current_summary, indent=2), encoding="utf-8"
        )
        (self.current_segment_dir / "speech_probs.json").write_text(
            json.dumps({"probs": self.current_probs}, indent=2), encoding="utf-8"
        )

        print(f"[SPEECH TRACKER] END → saved {self.current_segment_dir}\n")

        # reset for next segment
        self.current_segment_dir = None
        self.current_audio = np.empty(0, dtype=np.float32)
        self.current_probs = []
        self.current_start_frame = -1
