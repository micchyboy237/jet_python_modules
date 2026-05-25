import json
from pathlib import Path
from typing import List, Literal, Optional, Union

import matplotlib
import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechEndReason, SpeechSegment
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_PROB_WEIGHT,
    DEFAULT_RMS_WEIGHT,
    DEFAULT_THRESHOLD,
)
from jet.audio.audio_waveform.vad.vad_segment_scorer import save_vad_score
from jet.audio.helpers.config import (
    HOP_SIZE,
    HOP_STEP_S,
    SAMPLE_RATE,
)
from jet.audio.helpers.energy_base import compute_rms_per_frame
from jet.audio.helpers.loudness import normalize_loudness
from jet.audio.normalization.norm_speech_loudness import normalize_audio_for_vad
from jet.audio.speech.vad_types import ValleyTrough
from jet.audio.speech.wav_utils import save_wav_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    """
    if isinstance(probs, list):
        probs = np.asarray(probs, dtype=np.float32)
    elif not isinstance(probs, np.ndarray):
        raise TypeError("probs must be a list[float] or np.ndarray")

    n_frames = len(probs)
    if n_frames == 0:
        return np.array([], dtype=np.float32)

    n_audio_frames = len(audio_np) // frame_samples
    n_common = min(n_frames, n_audio_frames)
    if n_common == 0:
        return np.array([], dtype=np.float32)

    frames = audio_np[: n_common * frame_samples].reshape(n_common, frame_samples)
    rms_arr = np.sqrt(np.mean(frames**2, axis=1))
    rms_ceil = np.percentile(rms_arr, 99) + 1e-10
    rms_norm = np.clip(rms_arr / rms_ceil, 0.0, 1.0)

    hybrid = prob_weight * probs[:n_common] + rms_weight * rms_norm

    if n_frames > n_common:
        pad = prob_weight * probs[n_common:]
        hybrid = np.concatenate([hybrid, pad])

    return hybrid.astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers used by save_segment / save_segments
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
    speech_threshold: float = DEFAULT_THRESHOLD,
    is_already_hybrid: bool = True,
) -> None:
    """Save a speech-probability, RMS energy, and hybrid score plot."""
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
    if not is_already_hybrid:
        ax.axhline(
            y=speech_threshold,
            linestyle="--",
            color="#d62728",
            alpha=0.65,
            linewidth=1.2,
            label=f"threshold = {speech_threshold}",
        )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, num_frames - 1)
    ax.set_ylabel("Speech Probability", fontsize=10.5)
    ax.set_xlabel(
        f"Frame (10 ms)  —  {num_frames} frames ≈ {duration_sec:.1f} s", fontsize=10.5
    )
    ax.set_title(
        f"Segment {segment_idx:03d} — Model Probabilities", fontsize=12, pad=12
    )
    ax.grid(True, alpha=0.28, linestyle="--", zorder=0)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    ax_idx = 1

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

    if has_hybrid:
        ax_hyb = axes[ax_idx]
        n_hyb = len(hybrid)
        ax_hyb.plot(
            hybrid,
            color="#9467bd",
            linewidth=1.8,
            label=f"Hybrid score ({DEFAULT_PROB_WEIGHT}·prob + {DEFAULT_RMS_WEIGHT}·RMS)",
        )
        ax_hyb.fill_between(range(n_hyb), hybrid, color="#9467bd", alpha=0.14)
        if is_already_hybrid:
            ax_hyb.axhline(
                y=speech_threshold,
                linestyle="--",
                color="#d62728",
                alpha=0.65,
                linewidth=1.2,
                label=f"threshold = {speech_threshold}",
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


def _recover_raw_probs(
    hybrid: np.ndarray,
    rms: np.ndarray,
) -> np.ndarray:
    """Invert hybrid scores back to raw model probabilities."""
    n = min(len(hybrid), len(rms))
    rms_ceil = np.percentile(rms[:n], 99) + 1e-10
    rms_norm = np.clip(rms[:n] / rms_ceil, 0.0, 1.0)
    raw = np.clip(
        (hybrid[:n] - DEFAULT_RMS_WEIGHT * rms_norm) / DEFAULT_PROB_WEIGHT,
        0.0,
        1.0,
    )
    return raw.astype(np.float32)


# ---------------------------------------------------------------------------
# New reusable single-segment saver
# ---------------------------------------------------------------------------


def save_segment(
    meta: SpeechSegment,
    audio_np: np.ndarray,
    seg_dir: Path,
    is_already_hybrid: bool = True,
) -> Optional[SpeechSegment]:
    """
    Save a single speech segment to `seg_dir`.
    Returns the updated metadata on success, None on critical failure (e.g. WAV save).
    """
    seg_dir.mkdir(parents=True, exist_ok=True)
    idx = meta["num"]
    wav_path = seg_dir / "sound.wav"

    audio_loudness = normalize_loudness(audio_np, SAMPLE_RATE)
    audio_loudness_np = audio_loudness.normalized_data
    audio_loudness_stats = audio_loudness.get_stats()

    audio_np_norm, norm_vad_stats = normalize_audio_for_vad(audio_np, SAMPLE_RATE)

    audio_flat = np.asarray(audio_np, dtype=np.float32)
    if audio_flat.ndim == 2:
        audio_flat = audio_flat.mean(axis=1)
    elif audio_flat.ndim != 1:
        console.print(
            f"[red]Unexpected audio shape {audio_flat.shape}, skipping WAV save.[/red]"
        )
        return None

    wav_path = seg_dir / "sound.wav"
    seg_sound_file = save_wav_file(wav_path, audio_np)

    wav_path = seg_dir / "sound_norm_vad.wav"
    save_wav_file(wav_path, audio_np_norm)

    wav_path = seg_dir / "sound_louder.wav"
    save_wav_file(wav_path, audio_loudness_np)

    seg_probs_arr = np.asarray(meta["segment_probs"], dtype=np.float32)
    rms_list: List[float] = compute_rms_per_frame(audio_flat)
    rms = np.asarray(rms_list, dtype=np.float32)

    vad_score_path = save_vad_score(meta["segment_probs"], seg_dir, meta["num"])

    if is_already_hybrid:
        # segment_probs ARE the hybrid scores — use them directly for hybrid_probs.json.
        # Recover raw model probs for speech_probs.json (best-effort inversion).
        hybrid_arr = seg_probs_arr
        speech_probs_arr = _recover_raw_probs(hybrid=seg_probs_arr, rms=rms)
    else:
        # segment_probs are raw model probs — derive hybrid from them.
        speech_probs_arr = seg_probs_arr
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

    probs_info = {
        "num_frames": int(len(speech_probs_arr)),
        "mean": float(np.mean(speech_probs_arr)),
        "max": float(np.max(speech_probs_arr)),
        "min": float(np.min(speech_probs_arr)),
        "std": float(np.std(speech_probs_arr)),
        "median": float(np.median(speech_probs_arr)),
        "frame_rate_hz": 100,
    }

    meta_to_save = dict(meta)
    meta_to_save["output_path"] = str(wav_path.relative_to(seg_dir.parent.parent))
    meta_to_save["probs_info"] = {
        **probs_info,
        "frame_shift_sec": 0.010,
        "frame_start": meta.get("frame_start", 0),
        "recovered_from_hybrid": is_already_hybrid,
    }
    meta_to_save["energies_info"] = {
        "frame_shift_sec": 0.010,
        "num_frames": int(len(rms)),
    }
    if len(hybrid_arr) > 0:
        meta_to_save["hybrid_info"] = {
            "frame_shift_sec": 0.010,
            "frame_start": meta.get("frame_start", 0),
            "num_frames": int(len(hybrid_arr)),
            "threshold": DEFAULT_THRESHOLD,
        }
    meta_to_save.pop("segment_probs", None)
    meta_to_save.pop("energies", None)

    with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta_to_save, fh, indent=2, ensure_ascii=False)

    with open(seg_dir / "audio_loudness_stats.json", "w", encoding="utf-8") as fh:
        json.dump(audio_loudness_stats, fh, indent=2, ensure_ascii=False)

    with open(seg_dir / "norm_vad_stats.json", "w", encoding="utf-8") as fh:
        json.dump(norm_vad_stats, fh, indent=2, ensure_ascii=False)

    with open(seg_dir / "speech_probs.json", "w", encoding="utf-8") as fh:
        json.dump(speech_probs_arr.tolist(), fh, indent=2)

    with open(seg_dir / "energies.json", "w", encoding="utf-8") as fh:
        json.dump(rms.tolist(), fh, indent=2)

    if len(hybrid_arr) > 0:
        with open(seg_dir / "hybrid_probs.json", "w", encoding="utf-8") as fh:
            json.dump(hybrid_arr.tolist(), fh, indent=2)

    generate_plot(
        probs=speech_probs_arr,
        segment_idx=idx,
        duration_sec=float(meta["duration"]),
        output_path=seg_dir / "speech_and_rms.png",
        rms=rms,
        hybrid=hybrid_arr,
        speech_threshold=DEFAULT_THRESHOLD,
        is_already_hybrid=is_already_hybrid,
    )

    updated_meta = dict(meta)
    updated_meta["output_path"] = meta_to_save["output_path"]
    return updated_meta


# ---------------------------------------------------------------------------
# Multi-segment wrapper (unchanged public API)
# ---------------------------------------------------------------------------


def save_segments(
    segments: List[SpeechSegment],
    audio_chunks: List[np.ndarray],
    output_base_dir: Path,
    show_progress: bool = True,
    is_already_hybrid: bool = True,
) -> List[SpeechSegment]:
    """
    Persist every speech segment to output_base_dir/segments/segment_NNN/.
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_base_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    speech_segments = [s for s in segments if s["type"] == "speech"]

    if len(speech_segments) != len(audio_chunks):
        console.print(
            f"[yellow]save_segments: {len(speech_segments)} speech segments but "
            f"{len(audio_chunks)} audio chunks — zipping by position.[/yellow]"
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

            updated_meta = save_segment(
                meta=meta,
                audio_np=audio_np,
                seg_dir=seg_dir,
                is_already_hybrid=is_already_hybrid,
            )

            if updated_meta:
                saved.append(updated_meta)

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


# (make_segment left unchanged)
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
