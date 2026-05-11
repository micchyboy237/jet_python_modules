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
    normalize_energy,
    rms_to_loudness_label,
    smooth_signal,
)
from jet.audio.norm.norm_audio import normalize_audio
from jet.audio.speech.vad_extractors import (
    extract_valley_troughs,
    extract_valley_troughs_from_np_audio,
)
from jet.transformers.object import make_serializable
from rich.console import Console

console = Console()

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
        short = event.segment_dir.name
        console.print(
            f"[SAVER] Created directory → [bold blue link=file://{event.segment_dir}]{short}[/bold blue link=file://{event.segment_dir}]",
            highlight=False,
        )

    def on_segment_end(self, event: SpeechSegmentEndEvent) -> None:
        if event.segment_dir is None:
            console.print(
                "[SAVER] [yellow]Warning: no segment_dir → cannot save[/yellow]",
                highlight=False,
            )
            return

        dir_path = event.segment_dir
        dir_short = dir_path.name

        try:
            # 1. Audio - Save in full precision (32-bit float)
            wav_path = dir_path / "sound.wav"
            wav_short = wav_path.name
            if len(event.audio) > 0:
                try:
                    sf.write(
                        str(wav_path),
                        event.audio,
                        16000,
                        # subtype="FLOAT",  # ← Preserve float32
                        # subtype="PCM_32",  # Alternative: 32-bit integer
                    )
                    console.print(
                        f"  Saved [green link=file://{wav_path}]{wav_short}[/green link=file://{wav_path}] ([cyan]{len(event.audio):,}[/cyan] samples) [32-bit float]",
                        highlight=False,
                    )
                except Exception as e:
                    console.print(
                        f"  [red]Failed to save {wav_short}[/red]: {e}",
                        highlight=False,
                    )
            else:
                console.print(
                    "  [yellow]No audio data — skipping .wav[/yellow]", highlight=False
                )

            # 2. Summary
            speech_frames = sum(1 for f in event.prob_frames if f["is_speech"])
            speech_frames_pctg = (speech_frames / len(event.prob_frames)) * 100
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
                "speech_frames": speech_frames,
                "speech_frames_pctg": round(speech_frames_pctg, 1),
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
            raw_probs_short = raw_probs_path.name
            raw_probs_path.write_text(json.dumps(raw_probs, indent=2), encoding="utf-8")
            console.print(
                f"  Saved [green link=file://{raw_probs_path}]{raw_probs_short}[/green link=file://{raw_probs_path}] ([cyan]{len(raw_probs)}[/cyan] frames)",
                highlight=False,
            )

            # 3b. Save per-frame results with energy
            prob_frames_with_energy = self._compute_energies(
                event.audio, event.prob_frames
            )
            probs_path = dir_path / "speech_probs.json"
            probs_short = probs_path.name
            probs_path.write_text(
                json.dumps({"probs": prob_frames_with_energy}, indent=2),
                encoding="utf-8",
            )

            # No longer saving standalone energies.json; energies are included per-frame in speech_probs.json
            # The following block is deprecated and removed per refactor:
            # If needed, energies can be extracted from prob_frames_with_energy

            # 4. Plot
            self._generate_speech_prob_plot(event, prob_frames_with_energy)

            console.print(
                f"[SAVER] [bold green]Finished →[/bold green] [bold blue link=file://{dir_path}]{dir_short}[/bold blue link=file://{dir_path}]\n",
                highlight=False,
            )
        except Exception as e:
            console.print(
                f"[SAVER] [red]Failed writing file[/red]: {e}", highlight=False
            )

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
            # Use the RMS already captured at record time. Falls back to
            # audio-slice recomputation only if the field is absent (e.g. old
            # pre-roll frames from before this change).
            rms = frame.get("rms") or compute_rms(
                audio[
                    idx * frame_size : (
                        (idx + 1) * frame_size if idx < len(frames) - 1 else len(audio)
                    )
                ]
            )

            start = idx * frame_size
            end = (idx + 1) * frame_size if idx < len(frames) - 1 else len(audio)
            audio_frame = audio[start:end]

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
            console.print(
                "  [yellow]No prob frames → skipping plot[/yellow]", highlight=False
            )
            return
        dir_path = event.segment_dir
        # ── Prepare data ───────────────────────────────────────────────────────
        xs = np.array([p["frame_idx"] for p in prob_frames])
        smoothed_probs = np.array([p["smoothed_prob"] for p in prob_frames])

        # Prefer the pre-computed hybrid_prob stored in each SpeechFrame.
        # This is the value produced by the wrapper using the correct chunk
        # boundary and the running _peak_rms, so it's more accurate than
        # recomputing here.
        hybrid_probs_raw = np.array([p.get("hybrid_prob", 0.0) for p in prob_frames])
        # We still need norm_rms per frame to draw the energy panel.
        # Read rms from the frame; normalize against the segment maximum.
        rms_values = np.array([p.get("rms") or p["energy"]["rms"] for p in prob_frames])
        vad_code, vad_fg, vad_bg = _vad_badge(getattr(event, "vad_type", "fr"))
        # Adaptive normalization + smoothing
        norm_energy, norm_max = normalize_energy(
            rms_values, fallback_max=0.1, return_max=True
        )
        # Smooth using shared helper (window=5 frames ≈ 50 ms at 10 ms hop)
        SMOOTH_WINDOW = 5
        smoothed_norm_energy = smooth_signal(norm_energy, window=SMOOTH_WINDOW)

        # ── Hybrid signal: use pre-computed value ─────────────────────────────
        hybrid_probs = [float(v) for v in hybrid_probs_raw]
        smoothed_hybrid = smooth_signal(hybrid_probs_raw, window=SMOOTH_WINDOW)

        # Persist hybrid_probs.json alongside the other per-segment artefacts
        hybrid_path = dir_path / "hybrid_probs.json"
        hybrid_short = hybrid_path.name
        hybrid_path.write_text(
            json.dumps(
                hybrid_probs,
                indent=2,
            ),
            encoding="utf-8",
        )
        console.print(
            f"  Saved [green link=file://{hybrid_path}]{hybrid_short}[/green link=file://{hybrid_path}] ([cyan]{len(hybrid_probs)}[/cyan] values)",
            highlight=False,
        )

        ##### For Testing #####
        # Persist hybrid_probs_valley_troughs.json

        # Compute valley troughs based on hybrid_probs (probabilities, not waveform)
        # The idea is to extract where the hybrid speech probability signal finds its valleys/troughs.
        hybrid_probs_valley_troughs = extract_valley_troughs(
            hybrid_probs,
            min_trough_offset_s=2.0,  # This parameter may need adjusting as hybrid_probs is frame-level, not samples
            smoothing_window=20,
        )

        if hybrid_probs_valley_troughs:
            hybrid_probs_valley_troughs_path = (
                dir_path / "_hybrid_probs_valley_troughs.json"
            )
            hybrid_probs_valley_troughs_short = hybrid_probs_valley_troughs_path.name
            hybrid_probs_valley_troughs_path.write_text(
                json.dumps(hybrid_probs_valley_troughs, indent=2),
                encoding="utf-8",
            )
            console.print(
                f"  Saved [green link=file://{hybrid_probs_valley_troughs_path}]{hybrid_probs_valley_troughs_short}[/green link=file://{hybrid_probs_valley_troughs_path}] ([cyan]{len(hybrid_probs_valley_troughs)}[/cyan] entries)",
                highlight=False,
            )

        # Normalize audio
        norm_audio_np = normalize_audio(event.audio)
        norm_valley_troughs = extract_valley_troughs_from_np_audio(
            norm_audio_np,
            min_trough_offset_s=2.0,
        )

        # Persist norm_valley_troughs.json
        if norm_valley_troughs:
            norm_valley_troughs_path = dir_path / "_norm_valley_troughs.json"
            norm_valley_troughs_short = norm_valley_troughs_path.name
            norm_valley_troughs_path.write_text(
                json.dumps(norm_valley_troughs, indent=2),
                encoding="utf-8",
            )
            console.print(
                f"  Saved [green link=file://{norm_valley_troughs_path}]{norm_valley_troughs_short}[/green link=file://{norm_valley_troughs_path}] ([cyan]{len(norm_valley_troughs)}[/cyan] entries)",
                highlight=False,
            )
        ##### End For Testing #####

        # Persist valley_troughs.json
        if event.valley_troughs:
            valley_troughs_path = dir_path / "_valley_troughs.json"
            valley_troughs_short = valley_troughs_path.name
            valley_troughs_path.write_text(
                json.dumps(event.valley_troughs, indent=2),
                encoding="utf-8",
            )
            console.print(
                f"  Saved [green link=file://{valley_troughs_path}]{valley_troughs_short}[/green link=file://{valley_troughs_path}] ([cyan]{len(event.valley_troughs)}[/cyan] entries)",
                highlight=False,
            )

        # Persist last_trough.json
        if event.last_trough:
            last_trough_path = dir_path / "_last_trough.json"
            last_trough_short = last_trough_path.name
            last_trough_path.write_text(
                json.dumps(event.last_trough, indent=2),
                encoding="utf-8",
            )
            console.print(
                f"  Saved [green link=file://{last_trough_path}]{last_trough_short}[/green link=file://{last_trough_path}]",
                highlight=False,
            )

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
        chart_short = chart_path.name
        plt.savefig(chart_path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        console.print(
            f"  Saved chart: [green link=file://{chart_path}]{chart_short}[/green link=file://{chart_path}]",
            highlight=False,
        )
