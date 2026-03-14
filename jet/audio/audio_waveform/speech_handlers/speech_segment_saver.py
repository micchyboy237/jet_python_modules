# jet_python_modules/jet/audio/audio_waveform/speech_segment_saver.py
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import soundfile as sf
from jet.audio.audio_waveform.speech_events import (
    SpeechSegmentEndEvent,
    SpeechSegmentStartEvent,
)
from jet.audio.audio_waveform.speech_handlers.base import SpeechSegmentHandler
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
        summary_path = dir_path / "summary.json"
        summary_path.write_text(
            json.dumps(make_serializable(event.summary), indent=2), encoding="utf-8"
        )

        # 3. Probabilities
        probs_path = dir_path / "speech_probs.json"
        probs_path.write_text(
            json.dumps({"probs": event.probs}, indent=2), encoding="utf-8"
        )

        # 4. Plot
        if event.probs:
            fig, ax = plt.subplots(figsize=(10, 2.5), dpi=120)
            xs = [p["frame_idx"] for p in event.probs]
            ys = [p["smoothed_prob"] for p in event.probs]
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
