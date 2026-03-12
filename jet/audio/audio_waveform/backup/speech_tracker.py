# jet_python_modules/jet/audio/audio_waveform/speech_tracker.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from fireredvad.core.constants import FRAME_PER_SECONDS, SAMPLE_RATE
from fireredvad.core.stream_vad_postprocessor import StreamVadFrameResult
from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig


class StreamingSpeechTracker:
    """Real-time speech segment detector & saver."""

    def __init__(
        self,
        save_dir: str,
        min_speech_duration_sec: float = 0.3,
        min_silence_duration_sec: float = 0.2,
        max_speech_duration_sec: float = 10.0,
        vad: FireRedStreamVad | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = SAMPLE_RATE
        self.min_speech_duration_sec = max(0.1, min_speech_duration_sec)
        self.min_silence_duration_sec = max(0.1, min_silence_duration_sec)
        self.max_speech_duration_sec = max_speech_duration_sec
        self.min_speech_samples = int(self.min_speech_duration_sec * self.sample_rate)
        self.max_speech_samples = int(self.max_speech_duration_sec * self.sample_rate)
        self.vad = vad if vad is not None else self._create_vad()
        self.pre_buffer: np.ndarray = np.empty(0, dtype=np.float32)
        self.speech_buffer: np.ndarray = np.empty(0, dtype=np.float32)
        self.is_speaking = False
        self.segment_counter = 0
        self.segment_dir: Path | None = None
        self.current_start_frame = -1
        self.current_segment_probs: List[float] = []

    def _create_vad(self) -> FireRedStreamVad:
        model_dir = str(
            Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD")
            .expanduser()
            .resolve()
        )
        config = FireRedStreamVadConfig(
            use_gpu=False,
            speech_threshold=0.55,
            smooth_window_size=5,
            pad_start_frame=5,
            min_speech_frame=int(self.min_speech_duration_sec * FRAME_PER_SECONDS),
            max_speech_frame=int(self.max_speech_duration_sec * FRAME_PER_SECONDS),
            min_silence_frame=int(self.min_silence_duration_sec * FRAME_PER_SECONDS),
            chunk_max_frame=30000,
        )
        return FireRedStreamVad.from_pretrained(model_dir, config=config)

    def process_chunk(self, samples: np.ndarray) -> None:
        if len(samples) == 0:
            return
        norm = self._normalize_chunk(samples)
        self._update_pre_buffer(norm)
        results = self.vad.detect_chunk(norm)
        self._handle_results(results, norm)

    def _normalize_chunk(self, samples: np.ndarray) -> np.ndarray:
        chunk_max = np.max(np.abs(samples)) + 1e-10
        target = 0.30
        if chunk_max < 0.20:
            gain = min(target / chunk_max, 8.0)
            return samples * gain
        return samples

    def _update_pre_buffer(self, samples: np.ndarray) -> None:
        self.pre_buffer = (
            np.concatenate([self.pre_buffer, samples])
            if len(self.pre_buffer) > 0
            else samples.copy()
        )
        max_pre = int(1.5 * self.sample_rate)
        if len(self.pre_buffer) > max_pre:
            self.pre_buffer = self.pre_buffer[-max_pre:]

    def _handle_results(
        self, results: List[StreamVadFrameResult], current_chunk: np.ndarray
    ) -> None:
        # 1. Append to current (old) segment first – it owns this chunk
        if self.is_speaking:
            if len(self.speech_buffer) == 0:
                self.speech_buffer = self.pre_buffer.copy()
            else:
                self.speech_buffer = np.concatenate([self.speech_buffer, current_chunk])
            for r in results:
                self.current_segment_probs.append(r.smoothed_prob)

        # 2. Process transitions in order (end before start) so a VAD max-split inside the same chunk is not missed.
        started_this_chunk = False
        for r in results:
            if r.is_speech_end and self.is_speaking:
                self._end_segment(r)
            if r.is_speech_start and not self.is_speaking:
                self._start_segment(r)
                started_this_chunk = True

        # 3. New segment started this chunk (VAD split) → give it the current pre_buffer
        if started_this_chunk:
            self.speech_buffer = self.pre_buffer.copy()

        # 4. Safeguard force-cut (still needed for the mock-test case)
        if self.is_speaking and len(self.speech_buffer) >= self.max_speech_samples:
            self._force_end_max_speech()

    def _force_end_max_speech(self) -> None:
        dummy_result = StreamVadFrameResult(
            frame_idx=-1,
            is_speech=False,
            raw_prob=0.0,
            smoothed_prob=0.0,
            is_speech_end=True,
            speech_end_frame=self.current_start_frame
            + int(self.max_speech_duration_sec * FRAME_PER_SECONDS),
        )
        self._end_segment(dummy_result)

        # === FIX: also restart a new segment so the rest of the speech is not lost
        start_frame = dummy_result.speech_end_frame + 1
        dummy_start_result = StreamVadFrameResult(
            frame_idx=start_frame,
            is_speech=True,
            raw_prob=0.0,
            smoothed_prob=0.0,
            is_speech_start=True,
            speech_start_frame=start_frame,
        )
        self._start_segment(dummy_start_result)
        # init new buffer with current pre_buffer (includes the chunk that triggered the force)
        self.speech_buffer = self.pre_buffer.copy()

    def _start_segment(self, result: StreamVadFrameResult) -> None:
        self.segment_counter += 1
        self.segment_dir = self.save_dir / f"segment_{self.segment_counter:04d}"
        self.is_speaking = True
        self.current_start_frame = result.speech_start_frame

    def _end_segment(self, result: StreamVadFrameResult) -> None:
        buf_len = len(self.speech_buffer)
        if buf_len < self.min_speech_samples:
            self._cleanup()
            return
        self._save_files(result)
        self._cleanup()

    def _save_files(self, result: StreamVadFrameResult) -> None:
        assert self.segment_dir is not None
        self.segment_dir.mkdir(parents=True, exist_ok=True)
        audio = self.speech_buffer
        sf.write(str(self.segment_dir / "sound.wav"), audio, self.sample_rate)
        metadata = {
            "segment_id": self.segment_counter,
            "start_frame": self.current_start_frame,
            "end_frame": result.speech_end_frame,
            "start_sec": round((self.current_start_frame - 1) / FRAME_PER_SECONDS, 3),
            "end_sec": round((result.speech_end_frame - 1) / FRAME_PER_SECONDS, 3),
            "duration_sec": round(len(audio) / self.sample_rate, 3),
            "min_speech_duration_sec": self.min_speech_duration_sec,
            "min_silence_duration_sec": self.min_silence_duration_sec,
            "max_speech_duration_sec": self.max_speech_duration_sec,
            "prob_info": {
                "num_frames": len(self.current_segment_probs),
                "avg_smoothed_prob": round(
                    sum(self.current_segment_probs) / len(self.current_segment_probs), 3
                )
                if self.current_segment_probs
                else 0.0,
                "min_smoothed_prob": min(self.current_segment_probs)
                if self.current_segment_probs
                else 0.0,
                "max_smoothed_prob": max(self.current_segment_probs)
                if self.current_segment_probs
                else 0.0,
            },
        }
        with open(self.segment_dir / "segment.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        # Separate file with full array of smoothed probs for this segment
        with open(self.segment_dir / "speech_probs.json", "w", encoding="utf-8") as f:
            json.dump(self.current_segment_probs, f, indent=2)
        print(
            f"✅ SAVED segment_{self.segment_counter:04d} → {self.segment_dir} "
            f"({metadata['duration_sec']}s)"
        )

    def _cleanup(self) -> None:
        self.is_speaking = False
        self.speech_buffer = np.empty(0, dtype=np.float32)
        self.current_start_frame = -1
        self.segment_dir = None
        self.current_segment_probs = []

    def reset(self) -> None:
        self.vad.reset()
        self.pre_buffer = np.empty(0, dtype=np.float32)
        self._cleanup()
        self.segment_counter = 0
