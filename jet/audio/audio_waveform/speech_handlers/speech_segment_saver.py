# jet.audio.audio_waveform.speech_segment_saver

import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from jet.audio.audio_waveform.speech_events import (
    SpeechSegmentEndEvent,
    SpeechSegmentStartEvent,
)
from jet.audio.audio_waveform.speech_handlers.base import SpeechSegmentHandler
from jet.audio.audio_waveform.speech_types import SpeechFrame
from jet.audio.helpers.energy import (
    LoudnessLabel,
    compute_amplitude,
    compute_rms,
    has_sound,
    rms_to_loudness_label,
)
from jet.transformers.object import make_serializable


class SpeechFrameEnergy(TypedDict):
    has_sound: bool
    loudness: LoudnessLabel
    rms: float
    amp: float


class SpeechFrameWithEnergy(SpeechFrame):
    energy: SpeechFrameEnergy


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

        try:
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
            prob_frames_with_energy = self._compute_energies(
                event.audio, event.prob_frames
            )
            probs_path = dir_path / "speech_probs.json"
            probs_path.write_text(
                json.dumps({"probs": prob_frames_with_energy}, indent=2),
                encoding="utf-8",
            )

            # 4. Plot
            self._generate_speech_prob_plot(event, prob_frames_with_energy)

            print(f"[SAVER] Finished → {dir_path}\n")
        except Exception as e:
            print(f"[SAVER] Failed writing file: {e}")

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

    def _compute_energies(
        self, audio: np.ndarray, frames: list[SpeechFrame]
    ) -> list[SpeechFrameWithEnergy]:
        frames_with_energy: list[SpeechFrameWithEnergy] = []

        frame_size = int(len(audio) / len(frames)) if frames else 0

        for idx, frame in enumerate(frames):
            # frame["frame_idx"] is assumed to be ordered and sequential
            # Calculate the start and end indices for the audio chunk corresponding to this frame
            start = idx * frame_size
            end = (idx + 1) * frame_size if idx < len(frames) - 1 else len(audio)
            audio_frame = audio[start:end]
            rms = compute_rms(audio_frame)
            energy_info: SpeechFrameEnergy = {
                "has_sound": has_sound(audio_frame),
                "loudness": rms_to_loudness_label(rms),
                "rms": round(rms, 4),
                "amp": round(compute_amplitude(audio_frame), 4),
            }
            frames_with_energy.append(
                {
                    **frame,
                    "energy": energy_info,
                }
            )

        return frames_with_energy

    def _generate_speech_prob_plot(
        self, event: SpeechSegmentEndEvent, prob_frames: list[SpeechFrameWithEnergy]
    ):
        if not prob_frames:
            print("  No prob frames → skipping plot")
            return

        dir_path = event.segment_dir

        # ── Prepare data ───────────────────────────────────────────────────────
        xs = np.array([p["frame_idx"] for p in prob_frames])

        smoothed_probs = np.array([p["smoothed_prob"] for p in prob_frames])

        rms_values = np.array([p["energy"]["rms"] for p in prob_frames])

        # Normalize: rms ∈ [0.0, 0.1] → [0.0, 1.0], clip above 0.1
        MAX_RMS = 0.1
        norm_energy = np.clip(rms_values / MAX_RMS, 0.0, 1.0)

        # ── Create even smaller figure ────────────────────────────────────────
        fig = plt.figure(figsize=(9, 6.2), dpi=140)  # reduced from (10,7)
        fig.suptitle(
            f"Segment {event.segment_id}  •  {event.duration_sec:.1f}s  •  {len(prob_frames)} frames",
            fontsize=11,  # smaller suptitle
            y=0.975,
        )

        gs = fig.add_gridspec(2, 1, hspace=0.38)

        # Common x-limits
        x_min, x_max = xs[0], xs[-1]

        # ── Top: Smoothed probability ─────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(xs, smoothed_probs, color="#1f77b4", lw=1.4, label="smoothed prob")
        ax1.axhline(0.5, color="darkred", ls="--", lw=0.9, alpha=0.7, label="threshold")
        ax1.set_title("Smoothed Speech Probability", fontsize=10)
        ax1.set_ylabel("Probability", fontsize=9)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlim(x_min, x_max)
        ax1.grid(True, alpha=0.3, ls=":")
        ax1.legend(loc="upper right", fontsize=8)

        # ── Bottom: Normalized energy ─────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(
            xs, norm_energy, color="#ff7f0e", lw=1.3, label="norm. energy (rms → 0–1)"
        )
        ax2.axhline(0.5, color="darkred", ls="--", lw=0.9, alpha=0.7, label="0.5 ref")
        ax2.set_title("Normalized Energy (RMS clipped at 0.1)", fontsize=10)
        ax2.set_xlabel("Frame index", fontsize=9)
        ax2.set_ylabel("Normalized [0–1]", fontsize=9)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlim(x_min, x_max)
        ax2.grid(True, alpha=0.3, ls=":")
        ax2.legend(loc="upper right", fontsize=8)

        # ── Final styling & save ──────────────────────────────────────────────
        plt.tight_layout(
            rect=[0.06, 0.04, 0.94, 0.92]
        )  # tighter margins for compact feel
        chart_path = dir_path / "speech_prob_energy_2panel.png"
        plt.savefig(chart_path, bbox_inches="tight", dpi=160)
        plt.close(fig)

        print(f"  Saved chart: {chart_path.name}")
