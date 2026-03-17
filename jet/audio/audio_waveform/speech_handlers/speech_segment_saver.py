# jet_python_modules/jet/audio/audio_waveform/speech_segment_saver.py
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from jet.audio.audio_waveform.speech_events import (
    SpeechSegmentEndEvent,
    SpeechSegmentStartEvent,
)
from jet.audio.audio_waveform.speech_handlers.base import SpeechSegmentHandler
from jet.audio.audio_waveform.speech_types import SpeechFrame
from jet.transformers.object import make_serializable


class SpeechSegmentSaver(SpeechSegmentHandler):
    """
    Default handler: creates per-segment folders and saves
    - sound.wav
    - summary.json
    - speech_probs.json
    - speech_prob_plot.png
    """

    def __init__(self, base_save_dir: str | Path):
        self.base_dir = Path(base_save_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def on_segment_start(self, event: SpeechSegmentStartEvent) -> None:
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"segment_{now_str}_{event.segment_id:03d}"
        event.segment_dir = self.base_dir / dir_name
        event.segment_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SAVER] Created directory → {event.segment_dir}")

    def on_segment_end(self, event: SpeechSegmentEndEvent) -> None:
        if event.segment_dir is None:
            print("[SAVER] Warning: no segment_dir → cannot save")
            return

        dir_path = event.segment_dir

        # 1. Audio
        wav_path = dir_path / "sound.wav"
        if len(event.audio) > 0:
            sf.write(str(wav_path), event.audio, 16000)
            print(f"  Saved {wav_path.name} ({len(event.audio):,} samples)")
        else:
            print("  No audio data — skipping .wav")

        # 2. Summary
        summary = {
            "segment_id": event.segment_id,
            "start_frame": event.start_frame,
            "end_frame": event.end_frame,
            "start_time_sec": round(event.start_time_sec, 3),
            "end_time_sec": round(event.end_time_sec, 3),
            "duration_sec": round(event.duration_sec, 3),
            "audio_samples": len(event.audio),
            "prob_frames": len(event.prob_frames),
            "forced_split": event.forced_split,
            "trigger_reason": event.trigger_reason,
            "started_at": event.started_at,
            **self._compute_stats(event.audio, event.prob_frames),
        }
        summary_path = dir_path / "summary.json"
        summary_path.write_text(
            json.dumps(make_serializable(summary), indent=2), encoding="utf-8"
        )

        # 3. Probabilities
        probs_path = dir_path / "speech_probs.json"
        probs_path.write_text(
            json.dumps({"probs": event.prob_frames}, indent=2), encoding="utf-8"
        )

        # 4. Plot
        if event.prob_frames:
            fig, ax = plt.subplots(figsize=(10, 2.5), dpi=120)
            xs = [p["frame_idx"] for p in event.prob_frames]
            ys = [p["smoothed_prob"] for p in event.prob_frames]
            ax.plot(xs, ys, color="#1f77b4", lw=1.1, label="smoothed prob")
            ax.axhline(0.5, color="darkred", ls="--", alpha=0.5, label="threshold")
            ax.set_ylim(0, 1.05)
            ax.set_xlim(xs[0], xs[-1])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            ax.set_title(f"Segment {event.segment_id} | {event.duration_sec:.1f}s")
            chart_path = dir_path / "speech_prob_plot.png"
            plt.savefig(chart_path, bbox_inches="tight", dpi=140)
            plt.close(fig)
            print(f"  Saved chart: {chart_path.name}")

        print(f"[SAVER] Finished → {dir_path}\n")

    def _compute_stats(self, audio: np.ndarray, frames: list[SpeechFrame]):
        # Compute statistics
        if frames:
            probs = [p["smoothed_prob"] for p in frames]
            is_speech_list = [p["is_speech"] for p in frames]
            avg_prob = sum(probs) / len(probs)
            max_prob = max(probs)
            min_prob = min(probs)
            speech_ratio = sum(is_speech_list) / len(is_speech_list)
        else:
            avg_prob = max_prob = min_prob = speech_ratio = 0.0

        if len(audio) > 0:
            rms = np.sqrt(np.mean(audio**2))
            energy_db = 20 * np.log10(rms + 1e-10)
        else:
            energy_db = -float("inf")

        return {
            "avg_smoothed_prob": round(avg_prob, 3),
            "max_smoothed_prob": round(max_prob, 3),
            "min_smoothed_prob": round(min_prob, 3),
            "speech_frame_ratio": round(speech_ratio, 3),
            "energy_db_avg": round(energy_db, 2),
        }
