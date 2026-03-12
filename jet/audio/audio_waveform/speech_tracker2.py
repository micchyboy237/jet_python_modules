import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
from fireredvad.stream_vad import FireRedStreamVad, StreamVadFrameResult


@dataclass
class SpeechSegment:
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame: int
    duration_sec: float
    folder_name: str


class StreamingSpeechTracker:
    """
    Tracks speech segments in a streaming fashion and saves:
      - audio clips (sound.wav)
      - summary.json (metadata)
      - speech_probs.json (per-frame probabilities)

    in timestamped subfolders under save_dir/segment_YYYYMMDD_HHMMSS/
    """

    def __init__(
        self,
        save_dir: str,
        vad: FireRedStreamVad,
        min_speech_duration_sec: float = 0.3,
        min_silence_duration_sec: float = 0.2,
        max_speech_duration_sec: float = 12.0,
        sample_rate: int = 16000,
        frame_shift_sec: float = 0.01,  # typically 10 ms
    ):
        self.vad = vad  # FireRedStreamVad instance
        self.pending_audio = np.array([], dtype=np.float32)
        self.frame_length_samples = 400  # 25 ms @ 16 kHz
        self.frame_shift_samples = 160  # 10 ms @ 16 kHz

        self.save_dir = Path(save_dir).expanduser().resolve()
        self.min_speech_sec = min_speech_duration_sec
        self.min_silence_sec = min_silence_duration_sec
        self.max_speech_sec = max_speech_duration_sec
        self.sample_rate = sample_rate
        self.frame_shift_sec = frame_shift_sec

        # State
        self.current_audio: List[np.ndarray] = []  # list of 10 ms chunks
        self.current_probs: List[float] = []
        self.current_start_frame: Optional[int] = None
        self.total_frames: int = 0
        self.segments: List[SpeechSegment] = []

        self.min_speech_frames = int(round(self.min_speech_sec / self.frame_shift_sec))
        self.min_silence_frames = int(
            round(self.min_silence_sec / self.frame_shift_sec)
        )
        self.max_speech_frames = int(round(self.max_speech_sec / self.frame_shift_sec))

        self.in_speech = False
        self.silence_counter = 0
        self.speech_counter = 0

        os.makedirs(self.save_dir, exist_ok=True)

    def reset(self):
        """Clear current state (new session / new speaker)"""
        self.current_audio.clear()
        self.current_probs.clear()
        self.current_start_frame = None
        self.total_frames = 0
        self.segments.clear()
        self.in_speech = False
        self.silence_counter = 0
        self.speech_counter = 0

    def update(
        self,
        frame_audio: np.ndarray,  # shape (n_samples,) – usually 160 or 256 samples
        is_speech: bool,  # binary decision after VAD
        speech_prob: float,  # raw / smoothed probability [0–1]
    ) -> Optional[SpeechSegment]:
        """
        Feed one frame (typically 10 ms).
        Returns a completed SpeechSegment if one just ended, else None.
        """
        self.total_frames += 1
        self.current_audio.append(frame_audio.copy())
        self.current_probs.append(speech_prob)

        completed_segment = None

        if is_speech:
            self.silence_counter = 0

            if not self.in_speech:
                self.speech_counter += 1
                if self.speech_counter >= self.min_speech_frames:
                    self.in_speech = True
                    # Back-date start
                    self.current_start_frame = self.total_frames - self.speech_counter
            else:
                self.speech_counter += 1

                # ─── Force split on max duration ────────────────────────
                if self.speech_counter >= self.max_speech_frames:
                    completed_segment = self._close_current_segment(force=True)
                    # After forced split, continue accumulating for next segment
                    if completed_segment:
                        # Start new segment immediately (current frame is still speech)
                        self.current_start_frame = self.total_frames - 1
                        self.speech_counter = 1
                        self.in_speech = True

        else:
            self.speech_counter = 0

            if self.in_speech:
                self.silence_counter += 1
                if self.silence_counter >= self.min_silence_frames:
                    completed_segment = self._close_current_segment()
            # else: pure silence → limit buffer size
            elif len(self.current_audio) > self.max_speech_frames:
                drop = len(self.current_audio) - self.max_speech_frames // 2
                self.current_audio = self.current_audio[drop:]
                self.current_probs = self.current_probs[drop:]
                self.total_frames -= drop  # optional: keep absolute frame count correct

        if completed_segment:
            self.segments.append(completed_segment)

        return completed_segment

    def _close_current_segment(self, force: bool = False) -> Optional[SpeechSegment]:
        if self.current_start_frame is None:
            return None
        if not force and self.silence_counter >= self.min_silence_frames:
            end_frame = self.total_frames - 1 - self.silence_counter
        else:
            end_frame = self.total_frames - 1
        duration_frames = end_frame - self.current_start_frame + 1
        if duration_frames <= 0:
            self._reset_after_close()
            return None
        duration_sec = duration_frames * self.frame_shift_sec
        if not force and duration_sec < self.min_speech_sec:
            self._reset_after_close()
            return None

        # === FIXED: build continuous (de-overlapped) audio ===
        if not self.current_audio:
            self._reset_after_close()
            return None

        parts = [self.current_audio[0]]
        for frame in self.current_audio[1:]:
            new_part = frame[
                -self.frame_shift_samples :
            ]  # only the advancing 160 samples
            parts.append(new_part)
        continuous = np.concatenate(parts)

        start_idx_in_buffer = self.current_start_frame - (
            self.total_frames - len(self.current_audio)
        )
        if start_idx_in_buffer < 0:
            start_idx_in_buffer = 0

        start_sample = start_idx_in_buffer * self.frame_shift_samples
        end_sample = min(
            len(continuous),
            start_sample + duration_frames * self.frame_shift_samples,
        )
        segment_audio = continuous[start_sample:end_sample]

        # === rest of the method unchanged ===
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        segment_dir = self.save_dir / f"segment_{now_str}"
        segment_dir.mkdir(exist_ok=True)
        wav_path = segment_dir / "sound.wav"
        sf.write(wav_path, segment_audio, self.sample_rate, subtype="PCM_16")

        segment_probs = self.current_probs[
            start_idx_in_buffer : start_idx_in_buffer + duration_frames
        ]
        probs_path = segment_dir / "speech_probs.json"
        with open(probs_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "probs": [round(p, 4) for p in segment_probs],
                    "frame_shift_sec": self.frame_shift_sec,
                },
                f,
                indent=2,
            )
        summary = {
            "start_sec": round(self.current_start_frame * self.frame_shift_sec, 3),
            "end_sec": round(end_frame * self.frame_shift_sec, 3),
            "duration_sec": round(duration_sec, 3),
            "wav_file": "sound.wav",
            "probs_file": "speech_probs.json",
            "created_at": now_str,
            "forced_split": force,
        }
        summary_path = segment_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"✅ SAVED segment ({duration_sec:.2f}s) -> {wav_path}")

        self._reset_after_close()
        segment = SpeechSegment(
            start_sec=summary["start_sec"],
            end_sec=summary["end_sec"],
            start_frame=self.current_start_frame,
            end_frame=end_frame,
            duration_sec=summary["duration_sec"],
            folder_name=segment_dir.name,
        )
        return segment

    def _reset_after_close(self):
        """Keep minimal context after closing a segment"""
        self.current_start_frame = None
        self.in_speech = False
        self.silence_counter = 0
        self.speech_counter = 0
        # Keep last ~0.5–1 sec as possible context for next start
        keep_len = int(1.0 / self.frame_shift_sec)  # 100 frames for 0.01s
        if len(self.current_audio) > keep_len:
            self.current_audio = self.current_audio[-keep_len:]
            self.current_probs = self.current_probs[-keep_len:]

    def get_all_segments(self) -> List[SpeechSegment]:
        return self.segments

    def process_chunk(self, chunk: np.ndarray) -> List[SpeechSegment]:
        """
        Process incoming audio chunk of arbitrary length.
        Feeds overlapping 25 ms (400-sample) frames to FireRedVAD.
        Returns list of any segments completed during this call.
        """
        if len(chunk) == 0:
            return []

        self.pending_audio = np.concatenate(
            [self.pending_audio, chunk.astype(np.float32)]
        )

        completed_segments = []

        while len(self.pending_audio) >= self.frame_length_samples:
            # Take the next 400-sample frame
            frame = self.pending_audio[: self.frame_length_samples]

            # Run FireRedVAD
            result: StreamVadFrameResult = self.vad.detect_frame(frame)

            # Feed to segment tracker
            segment = self.update(
                frame_audio=frame,
                is_speech=result.is_speech,
                speech_prob=result.smoothed_prob,
            )

            if segment is not None:
                completed_segments.append(segment)

            # Advance by frame shift (10 ms = 160 samples) → overlapping
            self.pending_audio = self.pending_audio[self.frame_shift_samples :]

        return completed_segments

    # Optional: clean up on finalize
    def finalize(self) -> List[SpeechSegment]:
        # Discard remaining partial frame
        self.pending_audio = np.array([], dtype=np.float32)
        if self.in_speech:
            segment = self._close_current_segment(force=True)
            if segment:
                self.segments.append(segment)
        return self.segments
