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
from jet.audio.helpers.config import HOP_SIZE
from jet.audio.helpers.energy import (
    LoudnessLabel,
    compute_amplitude,
    compute_rms,
    has_sound,
    normalize_energy,
    rms_to_loudness_label,
    smooth_signal,
)
from jet.audio.helpers.energy_base import compute_rms_per_frame
from jet.transformers.object import make_serializable

# ── VAD badge metadata (mirrors live_srt_preview_handler) ───────────────────
_VAD_BADGE: dict[str, tuple[str, str, str]] = {
    #  key        code    face colour  box colour
    "fr": ("FRD", "#c084fc", "#3b1f6b"),  # purple — FireRed
    "silero": ("SIL", "#4ade80", "#1a3a2a"),  # green  — Silero
    "sb": ("SPB", "#60a5fa", "#1a2e4a"),  # blue   — SpeechBrain
    "ten_vad": ("TEN", "#fb923c", "#3a2210"),  # orange — TEN VAD
}


def _vad_badge(vad_type: str) -> tuple[str, str, str]:
    """Return (code, text_colour, box_colour) for matplotlib bbox annotation."""
    return _VAD_BADGE.get(vad_type, ("???", "#888888", "#2a2a2a"))


# ────────────────────────────────────────────────────────────────────────────


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
                "vad_type": event.vad_type,
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
                "segment_rms": event.segment_rms,
                "loudness": event.loudness,
                "has_sound": event.has_sound,
                "stats": event.stats,
                # **self._compute_stats(event.audio, event.prob_frames),
            }
            summary_path = dir_path / "summary.json"
            summary_path.write_text(
                json.dumps(make_serializable(summary), indent=2), encoding="utf-8"
            )

            # 3a. Save raw_probs in a separate probs.json
            if event.prob_frames and len(event.prob_frames) > 0:
                raw_probs = [p["raw_prob"] for p in event.prob_frames]
            else:
                raw_probs = []

            raw_probs_path = dir_path / "probs.json"
            raw_probs_path.write_text(json.dumps(raw_probs, indent=2), encoding="utf-8")
            print(f"  Saved {raw_probs_path.name} ({len(raw_probs)} frames)")

            # 3b. Save per-frame results with energy
            prob_frames_with_energy = self._compute_energies(
                event.audio, event.prob_frames
            )
            probs_path = dir_path / "speech_probs.json"
            probs_path.write_text(
                json.dumps({"probs": prob_frames_with_energy}, indent=2),
                encoding="utf-8",
            )

            # Save per-frame RMS energies using compute_rms_per_frame
            if len(event.audio) > 0 and event.prob_frames:
                hop_size = HOP_SIZE
                # We only have the short segment audio, so we must use
                # relative frame indices (start at 0). The old start_frame
                # and end_frame are global numbers from the live stream
                # and were causing every slice to be empty → all 0.0
                energies = compute_rms_per_frame(
                    audio=event.audio,
                    hop_size=hop_size,
                )

                energies_path = dir_path / "energies.json"
                energies_path.write_text(
                    json.dumps({"energies": energies}, indent=2), encoding="utf-8"
                )
                print(f"  Saved {energies_path.name} ({len(energies)} frames)")
            else:
                print("  No audio or frames — skipping energies.json")

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
            "stats": {
                "avg_smoothed_prob": round(avg_prob, 3),
                "max_smoothed_prob": round(max_prob, 3),
                "min_smoothed_prob": round(min_prob, 3),
                "speech_frame_ratio": round(speech_ratio, 3),
                "energy_db_avg": round(energy_db, 2),
            }
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
        vad_code, vad_fg, vad_bg = _vad_badge(getattr(event, "vad_type", "fr"))
        # Adaptive normalization + smoothing
        norm_energy, norm_max = normalize_energy(
            rms_values, fallback_max=0.1, return_max=True
        )
        # Smooth using shared helper (window=5 frames ≈ 50 ms at 10 ms hop)
        SMOOTH_WINDOW = 5
        smoothed_norm_energy = smooth_signal(norm_energy, window=SMOOTH_WINDOW)

        # ── Hybrid signal: same formula as the live visualizer ────────────────
        # hybrid[i] = 0.5 × smoothed_prob[i] + 0.5 × norm_rms[i]
        # Both operands are already computed above, so no new plumbing needed.
        hybrid_probs = 0.5 * smoothed_probs + 0.5 * norm_energy
        smoothed_hybrid = smooth_signal(hybrid_probs, window=SMOOTH_WINDOW)

        # Persist hybrid_probs.json alongside the other per-segment artefacts
        hybrid_path = dir_path / "hybrid_probs.json"
        hybrid_path.write_text(
            json.dumps(
                [float(v) for v in hybrid_probs],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  Saved {hybrid_path.name} ({len(hybrid_probs)} values)")

        # ── Figure: expand to 3 panels ────────────────────────────────────────
        fig = plt.figure(figsize=(9, 9.0), dpi=140)
        fig.suptitle(
            f"Segment {event.segment_id}  •  {event.duration_sec:.1f}s  •  "
            f"{len(prob_frames)} frames  •  VAD: {vad_code}  •  "
            f"max_rms={norm_max:.4f}  •  smooth_win={SMOOTH_WINDOW}",
            fontsize=11,
            y=0.975,
        )
        gs = fig.add_gridspec(3, 1, hspace=0.45)
        # Common x-limits
        x_min, x_max = xs[0], xs[-1]

        # ── Panel 1: Smoothed probability ─────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(xs, smoothed_probs, color="#1f77b4", lw=1.4, label="smoothed prob")
        ax1.axhline(0.5, color="darkred", ls="--", lw=0.9, alpha=0.7, label="threshold")
        # VAD badge — top-left corner of the probability panel
        ax1.text(
            0.01,
            0.97,
            vad_code,
            transform=ax1.transAxes,
            fontsize=8,
            fontfamily="monospace",
            fontweight="bold",
            color=vad_fg,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=vad_bg, edgecolor="none"),
        )
        ax1.set_title("Smoothed Speech Probability", fontsize=10)
        ax1.set_ylabel("Probability", fontsize=9)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlim(x_min, x_max)
        ax1.grid(True, alpha=0.3, ls=":")
        ax1.legend(loc="upper right", fontsize=8)

        # ── Panel 2: Normalized energy ────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(
            xs,
            norm_energy,
            color="#ff7f0e",
            lw=0.9,
            alpha=0.45,
            label="raw norm. energy",
        )
        ax2.plot(
            xs,
            smoothed_norm_energy,
            color="#d62728",
            lw=2.0,
            label=f"smoothed (win={SMOOTH_WINDOW})",
        )
        ax2.axhline(0.5, color="darkred", ls="--", lw=0.9, alpha=0.7, label="0.5 ref")
        ax2.set_title(
            f"Normalized Energy (adaptive max={norm_max:.4f}, smoothed)", fontsize=10
        )
        ax2.set_xlabel("Frame index", fontsize=9)
        ax2.set_ylabel("Normalized [0–1]", fontsize=9)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlim(x_min, x_max)
        ax2.grid(True, alpha=0.3, ls=":")
        ax2.legend(loc="upper right", fontsize=8)

        # ── Panel 3: Hybrid signal ────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(
            xs,
            hybrid_probs,
            color="#80cbc4",
            lw=0.9,
            alpha=0.45,
            label="raw hybrid",
        )
        ax3.plot(
            xs,
            smoothed_hybrid,
            color="#00897b",
            lw=2.0,
            label=f"smoothed (win={SMOOTH_WINDOW})",
        )
        ax3.axhline(
            0.5, color="darkred", ls="--", lw=0.9, alpha=0.7, label="0.5 threshold"
        )
        ax3.set_title("Hybrid Score  (0.5 × prob + 0.5 × norm_rms)", fontsize=10)
        ax3.set_xlabel("Frame index", fontsize=9)
        ax3.set_ylabel("Score [0–1]", fontsize=9)
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_xlim(x_min, x_max)
        ax3.grid(True, alpha=0.3, ls=":")
        ax3.legend(loc="upper right", fontsize=8)

        # ── Final styling & save ──────────────────────────────────────────────
        plt.tight_layout(rect=[0.06, 0.04, 0.94, 0.92])
        chart_path = dir_path / "speech_prob_energy_2panel.png"
        plt.savefig(chart_path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        print(f"  Saved chart: {chart_path.name}")
