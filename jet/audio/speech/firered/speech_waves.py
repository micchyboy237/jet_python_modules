from __future__ import annotations

import dataclasses
import json
import shutil
import statistics
from pathlib import Path
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from jet.audio.audio_types import AudioInput, MergedWaveInfo, SpeechWave
from jet.audio.helpers.config import HOP_SIZE, SAMPLE_RATE
from jet.audio.helpers.energy_base import (
    compute_rms_per_frame,
    normalize_energy,
)
from jet.audio.utils.loader import load_audio

WaveState = Literal["below", "above"]


@dataclasses.dataclass
class WaveShapeConfig:
    """
    Tunable thresholds that decide whether a probability wave has a real
    mountain shape rather than being a flat plateau or a tiny ripple.

    Attributes:
        min_prominence: How much the peak must rise above the average of the
            two surrounding valley endpoints.
        min_excursion: The minimum difference between the highest and lowest
            probability inside the wave window.
        min_peak_prob: Absolute floor — the peak frame must reach at least
            this probability (guards against waves that never really fire).
        min_frames: Waves shorter than this many frames are discarded.
        max_merge_gap_frames: If two consecutive raw waves are separated by a
            gap of this many frames or fewer, they are fused into one wave
            before shape validation. Set to 0 to disable merging entirely.
            At 10 ms/frame the default of 15 means gaps up to 150 ms are bridged.
        min_duration_sec: After merging, any wave whose duration is still
            shorter than this value (in seconds) is marked invalid and dropped.
            Default 0.08 s (80 ms) — roughly the shortest recognisable phoneme.
    """

    min_prominence: float = 0.05
    min_excursion: float = 0.04
    min_peak_prob: float = 0.55
    min_frames: int = 3
    max_merge_gap_frames: int = 15  # bridge gaps ≤ 150 ms (at 10 ms/frame)
    min_duration_sec: float = 0.08  # drop anything still shorter than 80 ms


def _recompute_wave_details(
    wave: SpeechWave,
    speech_probs: List[float],
    crossing: np.ndarray,
    sampling_rate: int,
    shape_cfg: WaveShapeConfig,
    open_threshold: float = 0.5,
    merge_count: int = 0,
) -> SpeechWave:
    """
    Recalculate all detail fields for *wave* using the global prob/hybrid arrays.

    Call this after adjusting frame_start / frame_end (e.g. after a merge) so
    that min/max/avg/std and shape diagnostics are always consistent with the
    actual frame boundaries stored in the wave.
    """
    frame_start = wave["details"]["frame_start"]
    frame_end = wave["details"]["frame_end"]
    frame_len = frame_end - frame_start

    wave_probs = speech_probs[frame_start:frame_end]
    wave_hybrid = list(crossing[frame_start:frame_end])

    entry_prob = speech_probs[frame_start - 1] if frame_start > 0 else 0.0
    exit_prob = speech_probs[frame_end] if frame_end < len(speech_probs) else 0.0

    shape_ok, shape_diag = is_prominent_wave(
        wave_probs, entry_prob, exit_prob, shape_cfg
    )

    duration_sec = frame_len * HOP_SIZE / sampling_rate
    start_sec = frame_start * HOP_SIZE / sampling_rate
    end_sec = frame_end * HOP_SIZE / sampling_rate

    # Compose merged/recombination diagnostics.
    merged = merge_count > 0

    wave["start_sec"] = start_sec
    wave["end_sec"] = end_sec
    wave["details"] = {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_len": frame_len,
        "duration_sec": duration_sec,
        "min_prob": min(wave_probs) if wave_probs else 0.0,
        "max_prob": max(wave_probs) if wave_probs else 0.0,
        "avg_prob": statistics.mean(wave_probs) if wave_probs else 0.0,
        "std_prob": statistics.stdev(wave_probs) if frame_len > 1 else 0.0,
        "avg_hybrid": float(np.mean(wave_hybrid)) if wave_hybrid else 0.0,
        # Count frames where the *hybrid* signal (not raw prob) stays above
        # the open threshold — this is what "holding above speech level" means.
        "rms_hold_frames": int(np.sum(np.asarray(wave_hybrid) >= open_threshold)),
        "merge_count": merge_count,
        "merged": merged,
        "merged_waves": [],  # populated by merge_raw_waves after absorption
        **shape_diag,
    }
    wave["is_valid"] = (
        wave["has_risen"]
        and wave["has_multi_passed"]
        and shape_ok
        and duration_sec >= shape_cfg.min_duration_sec
    )
    return wave


def merge_raw_waves(
    raw_waves: List[SpeechWave],
    speech_probs: List[float],
    crossing: np.ndarray,
    sampling_rate: int,
    shape_cfg: WaveShapeConfig,
    open_threshold: float = 0.5,
) -> List[SpeechWave]:
    """
    Fuse consecutive raw waves whose inter-wave gap is small.

    Why this helps
    --------------
    A single spoken word can produce two or three short probability bursts if
    the speaker takes a micro-breath or the VAD dips briefly between phonemes.
    Each burst alone may be too short to pass ``min_frames``, but the combined
    wave is long enough and carries all the speech data.

    Algorithm
    ---------
    1. Walk the list left-to-right.
    2. When the gap between wave[i].frame_end and wave[i+1].frame_start is
       ≤ max_merge_gap_frames, extend wave[i]'s frame_end to cover wave[i+1]
       and accumulate the merge count, then skip wave[i+1].
    3. After merging boundaries, call _recompute_wave_details so all stats
       (min/max/avg/shape) reflect the new, wider window.
    4. Return the merged list (may be shorter than the input).
    """
    if not raw_waves or shape_cfg.max_merge_gap_frames <= 0:
        return raw_waves

    merged: List[SpeechWave] = []
    current = raw_waves[0]
    merge_count = 0
    absorbed: List[MergedWaveInfo] = []

    for next_wave in raw_waves[1:]:
        gap = next_wave["details"]["frame_start"] - current["details"]["frame_end"]
        if gap <= shape_cfg.max_merge_gap_frames:
            # ── Absorb next_wave into current ──────────────────────────────
            # Extend the right boundary to the end of the next wave.
            current["details"]["frame_end"] = next_wave["details"]["frame_end"]
            # Inherit the "fallen" flag only if next wave had truly fallen.
            current["has_fallen"] = next_wave["has_fallen"]
            # A merged wave definitely crossed the threshold more than once.
            current["has_multi_passed"] = True
            merge_count += 1
            absorbed.append(
                MergedWaveInfo(
                    frame_start=next_wave["details"]["frame_start"],
                    frame_end=next_wave["details"]["frame_end"],
                    start_sec=next_wave["start_sec"],
                    end_sec=next_wave["end_sec"],
                    duration_sec=next_wave["details"].get("duration_sec", 0.0),
                    max_prob=next_wave["details"].get("max_prob", 0.0),
                    prominence=next_wave["details"].get("prominence", 0.0),
                )
            )
        else:
            # ── Finalise current and start fresh ───────────────────────────
            current = _recompute_wave_details(
                current,
                speech_probs,
                crossing,
                sampling_rate,
                shape_cfg,
                open_threshold,
                merge_count,
            )
            current["details"]["merged_waves"] = absorbed
            merged.append(current)
            current = next_wave
            merge_count = 0
            absorbed = []

    # Handle the last item in the chain.
    current = _recompute_wave_details(
        current,
        speech_probs,
        crossing,
        sampling_rate,
        shape_cfg,
        open_threshold,
        merge_count,
    )
    current["details"]["merged_waves"] = absorbed
    merged.append(current)
    return merged


def is_prominent_wave(
    wave_probs: List[float],
    entry_prob: float,
    exit_prob: float,
    cfg: WaveShapeConfig,
) -> tuple[bool, dict]:
    """
    Decide whether a slice of probabilities forms a genuine mountain shape.

    The algorithm:
      1. Baseline = average of entry_prob and exit_prob (the "ground level").
      2. Peak     = maximum probability inside the slice.
      3. Prominence = peak - baseline.
      4. Excursion  = max - min inside the slice (vertical range).

    Returns:
        (passed: bool, diagnostics: dict)
    """
    if not wave_probs:
        return False, {}

    peak_prob = max(wave_probs)
    min_prob = min(wave_probs)
    baseline = (entry_prob + exit_prob) / 2.0
    prominence = peak_prob - baseline
    excursion = peak_prob - min_prob
    n_frames = len(wave_probs)

    passed = (
        prominence >= cfg.min_prominence
        and excursion >= cfg.min_excursion
        and peak_prob >= cfg.min_peak_prob
        and n_frames >= cfg.min_frames
    )

    diagnostics = {
        "baseline": round(baseline, 6),
        "peak_prob": round(peak_prob, 6),
        "prominence": round(prominence, 6),
        "excursion": round(excursion, 6),
        "n_frames": n_frames,
        "shape_passed": passed,
    }
    return passed, diagnostics


def compute_hybrid_signal(
    speech_probs: List[float],
    rms_values: List[float],
    prob_weight: float = 0.5,
    rms_weight: float = 0.5,
) -> np.ndarray:
    """
    Combine speech probability and RMS energy into a single hybrid score
    per frame.

    Both inputs are brought to the same [0, 1] scale first:
      - speech_probs are already in [0, 1] from the VAD model.
      - rms_values are normalized against the loudest frame in the window.

    The result is a weighted average:
        hybrid[i] = prob_weight * prob[i] + rms_weight * norm_rms[i]

    Weights do NOT need to sum to 1.0, but doing so keeps the output in
    [0, 1], which makes the existing threshold (default 0.5) directly
    comparable to using probability alone.

    Args:
        speech_probs: Per-frame VAD probabilities from FireRedVAD.
        rms_values:   Per-frame RMS energy values (raw, not normalized).
        prob_weight:  How much the VAD probability contributes (default 0.5).
        rms_weight:   How much the RMS energy contributes (default 0.5).

    Returns:
        np.ndarray of hybrid scores, one per frame, length = min(len(probs), len(rms)).
    """
    min_len = min(len(speech_probs), len(rms_values))
    probs_arr = np.asarray(speech_probs[:min_len], dtype=np.float64)
    rms_arr = np.asarray(rms_values[:min_len], dtype=np.float64)
    norm_rms = normalize_energy(rms_arr, clip=True)  # → [0, 1]
    hybrid = prob_weight * probs_arr + rms_weight * norm_rms
    return hybrid


def get_speech_waves(
    audio: AudioInput,
    speech_probs: List[float],
    threshold: float = 0.5,
    close_threshold: Optional[float] = None,
    sampling_rate: int = SAMPLE_RATE,
    shape_cfg: Optional[WaveShapeConfig] = None,
    prob_weight: float = 0.5,
    rms_weight: float = 0.5,
) -> List[SpeechWave]:
    """
    Identify complete speech waves (rise → sustained high → fall) from FireRedVAD probabilities.

    prob_weight / rms_weight control the hybrid VAD+energy signal used for
    threshold crossing. Both default to 0.5 (equal blend). Set rms_weight=0
    to restore the original probability-only behaviour.

    close_threshold: If given, hysteresis is enabled: wave closes on a lower
    threshold than it opens. If not, uses threshold for both open/close.
    """
    audio_np, loaded_sr = load_audio(audio, sr=sampling_rate, mono=True)
    all_waves = check_speech_waves(
        speech_probs=speech_probs,
        threshold=threshold,
        close_threshold=close_threshold,
        sampling_rate=loaded_sr,
        shape_cfg=shape_cfg,
        audio_np=audio_np,
        prob_weight=prob_weight,
        rms_weight=rms_weight,
    )
    valid_waves: List[SpeechWave] = []
    for wave in all_waves:
        if wave.get("is_valid", False):
            valid_waves.append(wave)
    return valid_waves


def check_speech_waves(
    speech_probs: List[float],
    threshold: float = 0.5,
    close_threshold: Optional[float] = None,
    sampling_rate: int = SAMPLE_RATE,
    shape_cfg: Optional[WaveShapeConfig] = None,
    audio_np: Optional[np.ndarray] = None,
    prob_weight: float = 0.5,
    rms_weight: float = 0.5,
) -> List[SpeechWave]:
    """
    Analyze speech probabilities from FireRedVAD and return complete wave
    metadata.  Updated for 10 ms hop length (HOP_SIZE samples per frame).

    When audio_np is provided the function computes a hybrid signal that
    blends VAD probability with normalised RMS energy (controlled by
    prob_weight / rms_weight).  Wave boundaries (rise/fall) are decided on
    this hybrid signal instead of raw probability, making detection more
    robust against frames where the model is confident but the microphone
    captured almost no energy, or vice-versa.

    When audio_np is None the function falls back to probability-only mode
    (original behaviour).

    If close_threshold is specified, the signal must drop below this value
    to close — creating hysteresis behaviour for more stable endpointing.
    """

    # If no separate close threshold is given, use the same value as open.
    _close_threshold = close_threshold if close_threshold is not None else threshold

    # ──────────────────────────────────────────────────────────────
    # Phase 0: Parameter / input setup
    if shape_cfg is None:
        shape_cfg = WaveShapeConfig()

    if not speech_probs:
        return []

    # ── Build per-frame RMS and hybrid signal ─────────────────────────────
    n_frames = len(speech_probs)
    if audio_np is not None and len(audio_np) > 0:
        rms_all = compute_rms_per_frame(audio_np, HOP_SIZE, 0, n_frames - 1)
        hybrid_signal = compute_hybrid_signal(
            speech_probs, rms_all, prob_weight, rms_weight
        )
    else:
        # Fallback: treat pure probability as the hybrid signal
        rms_all = [0.0] * n_frames
        hybrid_signal = np.asarray(speech_probs, dtype=np.float64)

    crossing = hybrid_signal  # thresholding and window logic always uses this

    waves: List[SpeechWave] = []
    current_wave: Optional[SpeechWave] = None
    state: WaveState = "below"
    rise_frame_idx: Optional[int] = None

    for i, prob in enumerate(speech_probs):
        frame_time_sec = i * HOP_SIZE / sampling_rate
        hybrid_val = float(crossing[i]) if i < len(crossing) else prob

        if state == "below":
            # open: signal crosses UP past open_threshold
            if hybrid_val >= threshold:
                rise_frame_idx = i
                current_wave = SpeechWave(
                    has_risen=True,
                    has_multi_passed=False,
                    has_fallen=False,
                    is_valid=False,
                    start_sec=frame_time_sec,
                    end_sec=frame_time_sec,
                    details={
                        "frame_start": i,
                        "frame_end": i,
                        "frame_len": 0,
                        "duration_sec": 0.0,
                        "min_prob": prob,
                        "max_prob": prob,
                        "avg_prob": prob,
                        "std_prob": 0.0,
                        "avg_hybrid": hybrid_val,
                        "rms_hold_frames": 0,
                        "merge_count": 0,
                    },
                )
                state = "above"
        else:  # state == "above"
            if hybrid_val >= _close_threshold:
                # ── Signal is "alive" — anywhere at or above the close floor ──
                # This covers three sub-zones:
                #   a) hybrid_val >= threshold          → strongly above open level
                #   b) _close_threshold <= hybrid_val < threshold → hysteresis band
                # In all cases the wave is still open and counts as sustained.
                if current_wave is not None:
                    current_wave["has_multi_passed"] = True
            else:
                # ── Signal fell below the close threshold — end the wave ───────
                if current_wave is not None:
                    current_wave["has_fallen"] = True
                    frame_start = rise_frame_idx if rise_frame_idx is not None else 0
                    frame_end = i
                    # Only update frame indices in details; keep other fields
                    current_wave["details"]["frame_start"] = frame_start
                    current_wave["details"]["frame_end"] = frame_end
                    # Recompute all details cleanly for this wave
                    current_wave = _recompute_wave_details(
                        current_wave,
                        speech_probs,
                        crossing,
                        sampling_rate,
                        shape_cfg,
                        open_threshold=threshold,
                        merge_count=0,
                    )
                    waves.append(current_wave)
                current_wave = None
                rise_frame_idx = None
                state = "below"

            # else: still in hysteresis band — stay open, do nothing

    if current_wave is not None:
        current_wave["has_fallen"] = False
        current_wave["is_valid"] = False
        if rise_frame_idx is not None:
            frame_start = rise_frame_idx
            frame_end = len(speech_probs)
            # Only update frame indices in details; keep other fields
            current_wave["details"]["frame_start"] = frame_start
            current_wave["details"]["frame_end"] = frame_end
            # Tail waves haven't fallen — use threshold as a stand-in exit prob
            # so _recompute can still run is_prominent_wave consistently.
            # _recompute will also set start_sec / end_sec / duration_sec correctly.
            current_wave = _recompute_wave_details(
                current_wave,
                speech_probs,
                crossing,
                sampling_rate,
                shape_cfg,
                open_threshold=threshold,
                merge_count=0,
            )
            # Override is_valid: a wave that hasn't fallen is not valid yet.
            current_wave["is_valid"] = False
        waves.append(current_wave)

    # ── Phase 2: merge nearby raw waves ────────────────────────────────
    # Fuse pairs (or chains) of waves whose inter-wave gap is short enough,
    # then re-validate each merged wave.  This recovers speech data that
    # would otherwise be dropped as individual short waves.
    waves = merge_raw_waves(
        waves,
        speech_probs,
        crossing,
        sampling_rate,
        shape_cfg,
        open_threshold=threshold,
    )

    return waves


def save_wave_audio(
    audio_np: np.ndarray,
    sampling_rate: int,
    frame_start: int,
    frame_end: int,
    output_path: Path,
    hop_size: int = HOP_SIZE,
) -> None:
    """Extract and save audio chunk for a wave based on frame indices."""
    start_sample = frame_start * hop_size
    end_sample = (frame_end + 1) * hop_size
    wave_audio = audio_np[start_sample:end_sample]
    wavfile.write(output_path, sampling_rate, wave_audio)


def save_wave_plot(
    probs: List[float],
    rms_values: List[float],
    output_path: Path,
    wave_num: int,
    seg_num: int,
    prob_weight: float = 0.5,
    rms_weight: float = 0.5,
) -> None:
    """Create visualization plot with three panels:
      1. VAD speech probability
      2. Raw RMS energy
      3. Hybrid signal (weighted blend of the two)

    Handles potential length mismatches between probs and rms_values.
    """
    min_length = min(len(probs), len(rms_values))
    probs_aligned = probs[:min_length]
    rms_aligned = rms_values[:min_length]
    hybrid = compute_hybrid_signal(probs_aligned, rms_aligned, prob_weight, rms_weight)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    frames = np.arange(min_length)

    # ── Chart 1: VAD probability ──────────────────────────────────────────
    ax1.plot(frames, probs_aligned, color="blue", linewidth=1)
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Threshold")
    ax1.set_ylabel("VAD Probability")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Segment {seg_num:03d} - Wave {wave_num:03d} (Valid: {wave_num})")
    ax1.legend()

    # ── Chart 2: Raw RMS energy ───────────────────────────────────────────
    ax2.plot(frames, rms_aligned, color="green", linewidth=1)
    ax2.set_ylabel("RMS Energy")
    ax2.grid(True, alpha=0.3)

    # ── Chart 3: Hybrid signal ────────────────────────────────────────────
    ax3.plot(
        frames,
        hybrid,
        color="darkorange",
        linewidth=1.5,
        label=f"Hybrid (prob×{prob_weight} + rms×{rms_weight})",
    )
    ax3.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Threshold")
    ax3.set_xlabel("Frame Index (relative to wave)")
    ax3.set_ylabel("Hybrid Score")
    ax3.set_ylim(0, max(1.0, float(hybrid.max()) * 1.05) if len(hybrid) else 1.0)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_wave_data(
    wave: SpeechWave,
    audio_np: np.ndarray,
    speech_probs: List[float],
    sampling_rate: int,
    output_dir: Path,
    seg_num: int,
    wave_num: int,
    hop_size: int = HOP_SIZE,
    prob_weight: float = 0.5,
    rms_weight: float = 0.5,
) -> None:
    """Save all wave-related data to the specified directory."""
    wave_dir = output_dir / f"segment_{seg_num:03d}_wave_{wave_num:03d}"
    wave_dir.mkdir(parents=True, exist_ok=True)

    frame_start = wave["details"]["frame_start"]
    frame_end = wave["details"]["frame_end"]

    wav_path = wave_dir / "sound.wav"
    save_wave_audio(audio_np, sampling_rate, frame_start, frame_end, wav_path, hop_size)

    wave_probs = speech_probs[frame_start:frame_end]
    probs_path = wave_dir / "speech_probs.json"
    with open(probs_path, "w") as f:
        json.dump(wave_probs, f, indent=2)

    rms_values = compute_rms_per_frame(audio_np, hop_size, frame_start, frame_end)
    energies_path = wave_dir / "energies.json"
    with open(energies_path, "w") as f:
        json.dump(rms_values, f, indent=2)

    hybrid_values = list(
        compute_hybrid_signal(wave_probs, rms_values, prob_weight, rms_weight)
    )
    hybrid_path = wave_dir / "hybrid_signal.json"
    with open(hybrid_path, "w") as f:
        json.dump(hybrid_values, f, indent=2)

    wave_json_path = wave_dir / "wave.json"
    wave_copy = wave.copy()
    wave_copy["segment_num"] = seg_num
    wave_copy["wave_num"] = wave_num
    wave_copy["prob_weight"] = prob_weight
    wave_copy["rms_weight"] = rms_weight
    with open(wave_json_path, "w") as f:
        json.dump(wave_copy, f, indent=2)

    plot_path = wave_dir / "wave_plot.png"
    save_wave_plot(
        wave_probs,
        rms_values,
        plot_path,
        wave_num,
        seg_num,
        prob_weight=prob_weight,
        rms_weight=rms_weight,
    )


# ── Reporting helpers ──


def _find_parent_seg_num(frame_start: int, segments: list, default: int = 1) -> int:
    """
    Return the segment number whose frame range contains frame_start.
    Falls back to `default` (1-based, matching the save loop) when no segment matches.
    Using a shared helper ensures the save loop and _build_wave_report
    always produce the same directory name.
    """
    for seg in segments:
        if seg["frame_start"] <= frame_start <= seg["frame_end"]:
            return seg["num"]
    return default


def _build_wave_report(
    wave: SpeechWave,
    wave_idx: int,
    waves_dir: Path,
    segments: list,
) -> dict:
    """
    Flatten one SpeechWave into a clean, self-contained report dict.
    Used for both summary.json rows and top_5_waves.json entries.
    """
    frame_start = wave["details"]["frame_start"]
    parent_seg_num = _find_parent_seg_num(frame_start, segments, default=1)

    dir_name = f"segment_{parent_seg_num:03d}_wave_{wave_idx:03d}"
    wav_abs = (waves_dir / dir_name / "sound.wav").resolve()
    plot_abs = (waves_dir / dir_name / "wave_plot.png").resolve()
    short = _shorten_path(str(wav_abs))

    d = wave["details"]
    return {
        # ── identity ──────────────────────────────────────────────────
        "wave": wave_idx,
        "dir": dir_name,
        # ── timing ────────────────────────────────────────────────────
        "start_sec": round(wave["start_sec"], 4),
        "end_sec": round(wave["end_sec"], 4),
        "dur_sec": round(d["duration_sec"], 4),
        # ── Plot file ────────────────────────────────────────────────
        "plot_path": str(plot_abs),
        # ── audio file ────────────────────────────────────────────────
        "sound_short": short,
        "sound_path": str(wav_abs),
        # ── probability scores ────────────────────────────────────────
        "scores": {
            "min_prob": round(d["min_prob"], 6),
            "max_prob": round(d["max_prob"], 6),
            "avg_prob": round(d["avg_prob"], 6),
            "std_prob": round(d["std_prob"], 6),
            "baseline": round(d.get("baseline", 0.0), 6),
            "prominence": round(d.get("prominence", 0.0), 6),
            "excursion": round(d.get("excursion", 0.0), 6),
        },
    }


def _top5_reports(
    speech_waves: List[SpeechWave],
    waves_dir: Path,
    segments: list,
    duration_weight: float = 0.5,
) -> list[dict]:
    """
    Return the 5 waves with the highest composite score, already serialised
    as report dicts (not raw SpeechWave objects).
    Composite score = prominence * log(1 + duration_sec * duration_weight)
    This rewards waves that are both prominent and long, while the log scale
    prevents very long but flat waves from dominating short, sharp ones.
    Set duration_weight=0 to rank by prominence only (legacy behaviour).
    """
    import math

    indexed = list(enumerate(speech_waves, 1))  # [(1, wave), (2, wave), …]

    def _composite(wave):
        d = wave["details"]
        prominence = d.get("prominence", d["max_prob"])
        duration_sec = d.get("duration_sec", 0.0)
        return prominence * math.log1p(duration_sec * duration_weight)

    ranked = sorted(indexed, key=lambda iv: _composite(iv[1]), reverse=True)
    return [
        _build_wave_report(wave, idx, waves_dir, segments) for idx, wave in ranked[:5]
    ]


def build_summary_rows(
    speech_waves: List[SpeechWave],
    waves_dir: Path,
    segments: list,
) -> list[dict]:
    """
    Build a flat list of report dicts — one per valid wave — used for both
    the rich summary table and summary.json.
    """
    return [
        _build_wave_report(wave, idx, waves_dir, segments)
        for idx, wave in enumerate(speech_waves, 1)
    ]


def _shorten_path(path_str: str) -> str:
    """
    Show only the last 2 components of a path to keep the table columns narrow.
    E.g. segment_001_wave_003/sound.wav
    """
    parts = Path(path_str).parts
    if len(parts) <= 2:
        return path_str
    return "/".join(parts[-2:])


if __name__ == "__main__":
    import argparse

    from jet.audio.speech.firered.speech_timestamps_extractor import (
        extract_speech_timestamps,
    )
    from jet.file.utils import save_file
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    DEFAULT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"

    parser = argparse.ArgumentParser(
        description="Extract speech timestamps from audio using TEN VAD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_AUDIO,
        help=f"Input audio file path (default: {DEFAULT_AUDIO})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(OUTPUT_DIR),
        help=f"Output results dir (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.1, help="VAD probability threshold"
    )
    parser.add_argument(
        "-c",
        "--close-threshold",
        type=float,
        default=None,
        help="(Hysteresis) Threshold for wave close (default: same as open threshold)",
    )
    parser.add_argument(
        "-s", "--hop-size", type=int, default=160, help="Frame hop size in samples"
    )
    parser.add_argument(
        "--min-speech-duration",
        "-d",
        type=int,
        default=250,
        help="Minimum speech segment duration in ms",
    )
    parser.add_argument(
        "--min-silence-duration",
        "-g",
        type=int,
        default=100,
        help="Minimum silence duration in ms",
    )
    parser.add_argument(
        "--include-non-speech",
        "-n",
        action="store_true",
        help="Include non-speech segments",
    )
    parser.add_argument(
        "--prob-weight",
        type=float,
        default=0.5,
        help="Weight for VAD probability in hybrid signal (default 0.5)",
    )
    parser.add_argument(
        "--rms-weight",
        type=float,
        default=0.5,
        help="Weight for RMS energy in hybrid signal (default 0.5)",
    )
    args = parser.parse_args()

    segments, scores = extract_speech_timestamps(
        audio=args.input,
        include_non_speech=args.include_non_speech,
        threshold=args.threshold,
        min_speech_duration_sec=args.min_speech_duration / 1000,
        min_silence_duration_sec=args.min_silence_duration / 1000,
        # max_speech_duration_sec
        with_scores=True,
    )

    # Load audio for wave extraction
    audio_np, sr = load_audio(args.input, sr=SAMPLE_RATE, mono=True)

    speech_waves = get_speech_waves(
        args.input,
        scores,
        threshold=args.threshold,
        close_threshold=args.close_threshold,
        prob_weight=args.prob_weight,
        rms_weight=args.rms_weight,
    )

    # Save main JSON files
    save_file(segments, OUTPUT_DIR / "segments.json")
    save_file(scores, OUTPUT_DIR / "speech_probs.json")
    save_file(speech_waves, OUTPUT_DIR / "speech_waves.json")

    # Create waves directory and save individual wave files
    waves_dir = OUTPUT_DIR / "waves"
    waves_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"\n[bold]Generating files for {len(speech_waves)} valid speech waves...[/bold]"
    )

    for wave_idx, wave in enumerate(speech_waves, 1):
        wave_frame_start = wave["details"]["frame_start"]

        parent_seg_num = _find_parent_seg_num(wave_frame_start, segments, default=1)

        save_wave_data(
            wave=wave,
            audio_np=audio_np,
            speech_probs=scores,
            sampling_rate=sr,
            output_dir=waves_dir,
            seg_num=parent_seg_num,
            wave_num=wave_idx,
            hop_size=args.hop_size,
            prob_weight=args.prob_weight,
            rms_weight=args.rms_weight,
        )

    # ── summary table & JSON ──────────────────────────────────────────────────
    rows = build_summary_rows(speech_waves, waves_dir, segments)
    save_file(rows, OUTPUT_DIR / "summary.json")

    # ── top-5 waves (built after waves_dir exists and dirs are known) ─────────
    top5 = _top5_reports(speech_waves, waves_dir, segments)
    save_file(top5, OUTPUT_DIR / "top_5_waves.json")

    table = Table(
        title=f"Speech Waves Summary  ({len(rows)} valid waves)",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", justify="right", no_wrap=True)
    table.add_column("Dir", style="cyan", justify="left", no_wrap=True)
    table.add_column("Start (s)", style="white", justify="right", no_wrap=True)
    table.add_column("End (s)", style="white", justify="right", no_wrap=True)
    table.add_column("Dur (s)", style="yellow", justify="right", no_wrap=True)
    table.add_column("Prominence", style="magenta", justify="right", no_wrap=True)
    table.add_column("Peak prob", style="green", justify="right", no_wrap=True)
    table.add_column("Play", style="bright_cyan", justify="center", no_wrap=True)
    table.add_column("Sound", style="bright_black", justify="left")

    top5_dirs = {w["dir"] for w in top5}

    for r in rows:
        is_top5 = r["dir"] in top5_dirs
        row_style = "bold" if is_top5 else ""
        star = "★ " if is_top5 else "  "

        dir_cell = f"[link=file://{r['plot_path']}]{r['dir']}[/link]"
        sound_cell = f"[link=file://{r['sound_path']}]{r['sound_short']}[/link]"
        play_cell = f"[link=file://{r['sound_path']}]▶[/link]"

        table.add_row(
            f"{star}{r['wave']}",
            dir_cell,
            f"{r['start_sec']:.2f}",
            f"{r['end_sec']:.2f}",
            f"{r['dur_sec']:.2f}",
            f"{r['scores']['prominence']:.3f}",
            f"{r['scores']['max_prob']:.3f}",
            play_cell,
            sound_cell,
            style=row_style,
        )

    console.print()
    console.print(table)
    console.print()
    console.print(
        f"[bold green]✓[/bold green] All wave files saved under : [cyan]{waves_dir}[/cyan]"
    )
    console.print(
        f"[bold green]✓[/bold green] summary.json              : [cyan][link=file://{(OUTPUT_DIR / 'summary.json').resolve()}]{(OUTPUT_DIR / 'summary.json').resolve()}[/link][/cyan]"
    )
    console.print(
        f"[bold green]✓[/bold green] top_5_waves.json          : [cyan][link=file://{(OUTPUT_DIR / 'top_5_waves.json').resolve()}]{(OUTPUT_DIR / 'top_5_waves.json').resolve()}[/link][/cyan]"
    )
