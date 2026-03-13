# jet_python_modules/jet/audio/audio_waveform/speech_tracker.py
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from fireredvad.core.stream_vad_postprocessor import StreamVadFrameResult
from jet.transformers.object import make_serializable


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

        # Enhancement: track forced split, trigger reason, vad states for richer summary/insights
        self.current_forced_split = False
        self.current_trigger_reason = "silence"
        self.current_vad_states = []  # optional, can track VAD state sequence

        # Optional: reference to postprocessor (for accessing state, forced_split, reason)
        self.postprocessor = None  # to be assigned externally for advanced use

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
            entry = {
                "frame_idx": result.frame_idx,
                "raw_prob": result.raw_prob,
                "smoothed_prob": result.smoothed_prob,
                "is_speech": result.is_speech,
                "vad_state": self._get_vad_state_name(),  # optional
            }
            self.current_probs.append(entry)
            # Optional: record VAD state history
            # self.current_vad_states.append(self._get_vad_state_name())

    def _get_vad_state_name(self) -> str:
        # Ideally this provides the VAD state name at current frame
        # If postprocessor is injected, use it for true state reporting
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

        # Do NOT reset current_forced_split or current_trigger_reason here anymore —
        # keep value from previous end (if any). It will be overwritten only when a real end happens.
        self.current_vad_states = []

        print(f"[SPEECH TRACKER] START → {self.current_segment_dir}")

    def _end_segment(self, result: StreamVadFrameResult) -> None:
        if not self.is_speaking or self.current_segment_dir is None:
            return

        self.is_speaking = False

        end_frame = result.speech_end_frame
        end_time_sec = round((end_frame - 1) / 100.0, 3)

        # Ensure we have the most recent forced/reason values
        # (in case they were set in previous frames but not yet consumed)
        if self.postprocessor is not None:
            self.current_forced_split = getattr(
                self.postprocessor, "was_last_end_forced", False
            )
            self.current_trigger_reason = getattr(
                self.postprocessor, "last_split_reason", "unknown"
            )

        # ─── Compute extra statistics ───────────────────────────────────────
        if self.current_probs:
            probs = [p["smoothed_prob"] for p in self.current_probs]
            is_speech_list = [p["is_speech"] for p in self.current_probs]

            avg_prob = round(sum(probs) / len(probs), 3)
            max_prob = round(max(probs), 3)
            min_prob = round(min(probs), 3)
            speech_ratio = round(sum(is_speech_list) / len(is_speech_list), 3)
        else:
            avg_prob = max_prob = min_prob = speech_ratio = 0.0

        # Energy (RMS) in dB
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
                # "vad_state_sequence": "".join(self.current_vad_states),  # optional
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
            json.dumps(make_serializable(self.current_summary), indent=2),
            encoding="utf-8",
        )
        (self.current_segment_dir / "speech_probs.json").write_text(
            json.dumps({"probs": self.current_probs}, indent=2), encoding="utf-8"
        )

        print(f"[SPEECH TRACKER] END → saved {self.current_segment_dir}\n")

        # Generate speech probs chart
        self._generate_speech_probs_chart()

        # Reset only AFTER saving — so next segment starts clean
        self.current_forced_split = False
        self.current_trigger_reason = "silence"  # or "unknown"
        self.current_segment_dir = None
        self.current_audio = np.empty(0, dtype=np.float32)
        self.current_probs = []
        self.current_start_frame = -1
        self.current_vad_states = []

    def _generate_speech_probs_chart(self):
        if self.current_probs:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 2.5), dpi=120)
            xs = [p["frame_idx"] for p in self.current_probs]
            ys = [p["smoothed_prob"] for p in self.current_probs]
            ax.plot(xs, ys, color="#1f77b4", lw=1.1, label="smoothed prob")
            ax.axhline(0.5, color="darkred", ls="--", alpha=0.5, label="threshold")
            ax.set_ylim(0, 1.05)
            ax.set_xlim(xs[0], xs[-1])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            ax.set_title(
                f"Segment {self.segment_counter}  |  {self.current_summary['duration_sec']:.1f}s",
                fontsize=11,
            )

            chart_path = self.current_segment_dir / "speech_prob_plot.png"
            plt.savefig(chart_path, bbox_inches="tight", dpi=140)
            plt.close(fig)
            print(f"  Saved probability chart: {chart_path.name}")
