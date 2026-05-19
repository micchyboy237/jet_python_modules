import json
from pathlib import Path
from typing import List, Literal, Optional, Union

import matplotlib
import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechEndReason, SpeechSegment
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_PROB_WEIGHT,
    DEFAULT_RMS_WEIGHT,
)
from jet.audio.helpers.config import (
    HOP_SIZE,
    HOP_STEP_S,
    SAMPLE_RATE,
)
from jet.audio.helpers.energy_base import compute_rms_per_frame
from jet.audio.speech.vad_types import ValleyTrough

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchaudio
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_PREROLL_HYBRID_THRESHOLD,
)
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()


# ---------------------------------------------------------------------------
# Reusable hybrid probability computation
# ---------------------------------------------------------------------------


def compute_hybrid_probs(
    probs: Union[List[float], np.ndarray],
    audio_np: np.ndarray,
    prob_weight: float = DEFAULT_PROB_WEIGHT,
    rms_weight: float = DEFAULT_RMS_WEIGHT,
    frame_samples: int = HOP_SIZE,
) -> np.ndarray:
    """
    Compute hybrid scores by combining speech probabilities with normalised RMS energy.

    The hybrid score per frame is:
        score = prob_weight * smoothed_prob + rms_weight * rms_norm

    RMS is normalised using the 99th-percentile of the segment's RMS values.

    Args:
        probs:         Speech probability values (one per 10ms frame).
                       Accepts either a Python list of floats or a numpy array.
        audio_np:      Corresponding audio signal as numpy array.
        prob_weight:   Weight for the speech probability component.
        rms_weight:    Weight for the RMS energy component.
        frame_samples: Number of audio samples per frame (160 @ 16kHz = 10ms).

    Returns:
        Numpy array of hybrid scores, same length as probs.
    """
    # Convert probs to numpy array if it's a list
    if isinstance(probs, list):
        probs = np.asarray(probs, dtype=np.float32)
    elif not isinstance(probs, np.ndarray):
        raise TypeError("probs must be a list[float] or np.ndarray")

    n_frames = len(probs)
    if n_frames == 0:
        return np.array([], dtype=np.float32)

    # Compute per-frame RMS aligned to probs
    n_audio_frames = len(audio_np) // frame_samples
    n_common = min(n_frames, n_audio_frames)
    if n_common == 0:
        return np.array([], dtype=np.float32)

    frames = audio_np[: n_common * frame_samples].reshape(n_common, frame_samples)
    rms_arr = np.sqrt(np.mean(frames**2, axis=1))
    rms_ceil = np.percentile(rms_arr, 99) + 1e-10
    rms_norm = np.clip(rms_arr / rms_ceil, 0.0, 1.0)

    hybrid = prob_weight * probs[:n_common] + rms_weight * rms_norm

    # If probs is longer than available audio frames, pad with prob-only values
    if n_frames > n_common:
        pad = prob_weight * probs[n_common:]
        hybrid = np.concatenate([hybrid, pad])

    return hybrid.astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers used by save_segments
# ---------------------------------------------------------------------------


def _frames_from_seconds(sec: float) -> int:
    """Convert seconds to a 10 ms frame index (100 frames per second)."""
    return int(round(sec * 100.0))


def generate_plot(
    probs: np.ndarray,
    segment_idx: int,
    duration_sec: float,
    output_path: Path,
    rms: Optional[np.ndarray] = None,
    hybrid: Optional[np.ndarray] = None,
    hybrid_threshold: float = DEFAULT_PREROLL_HYBRID_THRESHOLD,
) -> None:
    """Save a speech-probability, RMS energy, and hybrid score plot to *output_path*."""
    num_frames = len(probs)
    if num_frames == 0:
        return

    has_rms = rms is not None and len(rms) > 0
    has_hybrid = hybrid is not None and len(hybrid) > 0
    rows = 1 + int(has_rms) + int(has_hybrid)
    fig, axes = plt.subplots(rows, 1, figsize=(9.5, 3.2 * rows), dpi=140)
    if rows == 1:
        axes = [axes]

    # ── Row 0: raw speech probabilities ──────────────────────────────────
    ax = axes[0]
    ax.plot(probs, color="#2ca02c", linewidth=1.8, label="Speech probability")
    ax.fill_between(range(num_frames), probs, color="#2ca02c", alpha=0.14)
    ax.axhline(
        y=0.4,
        linestyle="--",
        color="#d62728",
        alpha=0.65,
        linewidth=1.2,
        label="threshold ≈ 0.4",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, num_frames - 1)
    ax.set_ylabel("Speech Probability", fontsize=10.5)
    ax.set_xlabel(
        f"Frame (10 ms)  —  {num_frames} frames ≈ {duration_sec:.1f} s",
        fontsize=10.5,
    )
    ax.set_title(
        f"Segment {segment_idx:03d} — Model Probabilities",
        fontsize=12,
        pad=12,
    )
    ax.grid(True, alpha=0.28, linestyle="--", zorder=0)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    ax_idx = 1

    # ── Row 1: RMS energy ─────────────────────────────────────────────────
    if has_rms:
        ax_rms = axes[ax_idx]
        ax_rms.plot(range(len(rms)), rms, linewidth=1.6, label="RMS energy")
        ax_rms.fill_between(range(len(rms)), rms, alpha=0.15)
        ax_rms.set_ylabel("RMS Energy", fontsize=10.5)
        ax_rms.set_xlabel("Frame (10 ms)", fontsize=10.5)
        ax_rms.set_xlim(0, len(rms) - 1)
        ax_rms.grid(True, alpha=0.28, linestyle="--", zorder=0)
        ax_rms.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
        ax_idx += 1

    # ── Row 2: hybrid score ───────────────────────────────────────────────
    if has_hybrid:
        ax_hyb = axes[ax_idx]
        n_hyb = len(hybrid)
        ax_hyb.plot(
            hybrid,
            color="#9467bd",
            linewidth=1.8,
            label="Hybrid score (0.5·prob + 0.5·RMS)",
        )
        ax_hyb.fill_between(range(n_hyb), hybrid, color="#9467bd", alpha=0.14)
        ax_hyb.axhline(
            y=hybrid_threshold,
            linestyle="--",
            color="#d62728",
            alpha=0.65,
            linewidth=1.2,
            label=f"threshold = {hybrid_threshold}",
        )
        ax_hyb.set_ylim(-0.03, 1.03)
        ax_hyb.set_xlim(0, n_hyb - 1)
        ax_hyb.set_ylabel("Hybrid Score", fontsize=10.5)
        ax_hyb.set_xlabel("Frame (10 ms)", fontsize=10.5)
        ax_hyb.grid(True, alpha=0.28, linestyle="--", zorder=0)
        ax_hyb.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    fig.tight_layout(pad=0.9)
    plt.savefig(output_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# save_segments
# ---------------------------------------------------------------------------


def save_segments(
    segments: List[SpeechSegment],
    audio_chunks: List[np.ndarray],
    output_base_dir: Path,
    show_progress: bool = True,
    is_already_hybrid: bool = True,
) -> List[SpeechSegment]:
    """
    Persist every speech segment to *output_base_dir/segments/segment_NNN/*.

    For each segment the function writes:
      sound.wav          – 16-kHz PCM-16 audio
      meta.json          – SpeechSegment metadata + probs_info summary
      speech_probs.json  – per-frame probabilities + summary stats
      energies.json      – per-frame RMS energy
      speech_and_rms.png – probability + RMS energy plot

    Args:
        is_already_hybrid: If True, segment_probs contains hybrid scores
                           (0.5·prob + 0.5·rms_norm) rather than raw model
                           probabilities. Raw probs will be recovered via
                           inversion before saving to speech_probs.json.
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_base_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    speech_segments = [s for s in segments if s["type"] == "speech"]

    if len(speech_segments) != len(audio_chunks):
        console.print(
            f"[yellow]save_segments: {len(speech_segments)} speech segments but "
            f"{len(audio_chunks)} audio chunks — zipping by position, extras ignored.[/yellow]"
        )

    pairs = list(zip(speech_segments, audio_chunks))
    saved: List[SpeechSegment] = []

    _progress: Optional[Progress] = None
    if show_progress and len(pairs) > 0:
        _progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

    def _run_saves() -> None:
        task = (
            _progress.add_task("[cyan]Saving segments + plots…", total=len(pairs))
            if _progress
            else None
        )

        for meta, audio_np in pairs:
            idx = meta["num"]
            seg_dir = segments_dir / f"segment_{idx:03d}"
            seg_dir.mkdir(exist_ok=True)

            # ── 1. WAV ────────────────────────────────────────────────────
            wav_path = seg_dir / "sound.wav"
            try:
                torchaudio.save(
                    str(wav_path),
                    torch.from_numpy(audio_np).unsqueeze(0),
                    SAMPLE_RATE,
                    encoding="PCM_S",
                    bits_per_sample=16,
                )
            except Exception as exc:
                console.print(f"[red]Failed to save WAV {wav_path}: {exc}[/red]")
                if _progress and task is not None:
                    _progress.advance(task)
                continue

            # ── 2. Probability array ──────────────────────────────────────
            seg_probs_arr = np.asarray(meta["segment_probs"], dtype=np.float32)

            # ── 3. RMS energies from segment (fallback: recompute from audio) ──
            rms_list: List[float] = compute_rms_per_frame(audio_np)
            rms = np.asarray(rms_list, dtype=np.float32)

            # ── 4. Recover raw probs if segment_probs are hybrid scores ───
            if is_already_hybrid:
                seg_probs_arr = _recover_raw_probs(hybrid=seg_probs_arr, rms=rms)

            # ── 5. probs_info summary stats ───────────────────────────────
            probs_info = {
                "num_frames": int(len(seg_probs_arr)),
                "mean": float(np.mean(seg_probs_arr)),
                "max": float(np.max(seg_probs_arr)),
                "min": float(np.min(seg_probs_arr)),
                "std": float(np.std(seg_probs_arr)),
                "median": float(np.median(seg_probs_arr)),
                "frame_rate_hz": 100,
            }

            # ── 6. meta.json ──────────────────────────────────────────────
            meta_to_save = dict(meta)
            meta_to_save["output_path"] = str(wav_path.relative_to(output_base_dir))
            meta_to_save["probs_info"] = probs_info
            meta_to_save.pop("segment_probs", None)
            meta_to_save.pop("energies", None)
            with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
                json.dump(meta_to_save, fh, indent=2, ensure_ascii=False)

            # ── 7. speech_probs.json ──────────────────────────────────────
            with open(seg_dir / "speech_probs.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "probs": seg_probs_arr.tolist(),
                        "frame_shift_sec": 0.010,
                        "frame_start": meta.get("frame_start", 0),
                        "summary": probs_info,
                        "recovered_from_hybrid": is_already_hybrid,
                    },
                    fh,
                    indent=2,
                )

            # ── 8. energies.json ──────────────────────────────────────────
            with open(seg_dir / "energies.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "rms": rms.tolist(),
                        "frame_shift_sec": 0.010,
                        "num_frames": int(len(rms)),
                    },
                    fh,
                    indent=2,
                )

            # ── 9. hybrid_probs.json ──────────────────────────────────────
            n_min = min(len(seg_probs_arr), len(rms))
            if n_min > 0:
                rms_ceil = np.percentile(rms[:n_min], 99) + 1e-10
                rms_norm = np.clip(rms[:n_min] / rms_ceil, 0.0, 1.0)
                hybrid_arr = (
                    DEFAULT_PROB_WEIGHT * seg_probs_arr[:n_min]
                    + DEFAULT_RMS_WEIGHT * rms_norm
                ).astype(np.float32)
            else:
                hybrid_arr = np.array([], dtype=np.float32)

            if len(hybrid_arr) > 0:
                with open(seg_dir / "hybrid_probs.json", "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "hybrid": hybrid_arr.tolist(),
                            "frame_shift_sec": 0.010,
                            "frame_start": meta.get("frame_start", 0),
                            "num_frames": int(len(hybrid_arr)),
                            "threshold": DEFAULT_PREROLL_HYBRID_THRESHOLD,
                        },
                        fh,
                        indent=2,
                    )

            # ── 10. speech_and_rms.png ─────────────────────────────────────
            generate_plot(
                probs=seg_probs_arr,
                segment_idx=idx,
                duration_sec=float(meta["duration"]),
                output_path=seg_dir / "speech_and_rms.png",
                rms=rms,
                hybrid=hybrid_arr,
                hybrid_threshold=DEFAULT_PREROLL_HYBRID_THRESHOLD,
            )

            meta["output_path"] = meta_to_save["output_path"]
            saved.append(meta)
            if _progress and task is not None:
                _progress.advance(task)

    if _progress:
        with _progress:
            _run_saves()
    else:
        _run_saves()

    console.print(f"[bold green]✓ Saved {len(saved)} segments[/bold green]")
    console.print(
        f"Output: [link=file://{segments_dir.resolve()}]{segments_dir}[/link]"
    )
    return saved


def _recover_raw_probs(
    hybrid: np.ndarray,
    rms: np.ndarray,
) -> np.ndarray:
    """
    Invert the hybrid score back to raw model probabilities.

    The hybrid formula is:
        hybrid = DEFAULT_PROB_WEIGHT * raw_prob + DEFAULT_RMS_WEIGHT * rms_norm
    Solving for raw_prob:
        raw_prob = (hybrid - DEFAULT_RMS_WEIGHT * rms_norm) / DEFAULT_PROB_WEIGHT

    Args:
        hybrid: Per-frame hybrid scores stored in segment_probs.
        rms:    Raw (unnormalised) RMS energies from segment["energies"].

    Returns:
        Recovered raw probabilities clipped to [0, 1].
    """
    n = min(len(hybrid), len(rms))
    rms_ceil = np.percentile(rms[:n], 99) + 1e-10
    rms_norm = np.clip(rms[:n] / rms_ceil, 0.0, 1.0)
    raw = np.clip(
        (hybrid[:n] - DEFAULT_RMS_WEIGHT * rms_norm) / DEFAULT_PROB_WEIGHT,
        0.0,
        1.0,
    )
    return raw.astype(np.float32)


def make_segment(
    num: int,
    start_sec: float,
    end_sec: float,
    probs: List[float],
    seg_type: Literal["speech", "non-speech"] = "speech",
    end_reason: "SpeechEndReason | None" = None,
    is_ongoing: bool = False,
    last_non_speech_sec: Optional[float] = None,
    best_valley_trough: Optional["ValleyTrough"] = None,
    sample_rate: int = SAMPLE_RATE,
    return_seconds: bool = False,
    with_scores: bool = False,
) -> SpeechSegment:
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    frame_start = int(start_sec / HOP_STEP_S)
    frame_end = int(end_sec / HOP_STEP_S)
    avg_prob = float(np.mean(probs))
    duration_sec = end_sec - start_sec
    start_val = start_sec if return_seconds else start_sample
    end_val = end_sec if return_seconds else end_sample
    return SpeechSegment(
        num=num,
        start=start_val,
        end=end_val,
        duration=duration_sec,
        end_reason=end_reason,
        is_ongoing=is_ongoing,
        last_non_speech_sec=last_non_speech_sec,
        best_valley_trough=best_valley_trough,
        prob=avg_prob,
        frames_length=len(probs),
        frame_start=frame_start,
        frame_end=frame_end,
        type=seg_type,
        segment_probs=probs,
    )
