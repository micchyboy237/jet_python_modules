"""
Silero VAD Analyzer – One-click full insight generator
Features:
- Loads any audio file
- Runs Silero VAD with your chosen threshold
- Generates:
    • Interactive probability plot (with threshold lines)
    • Speech segments timeline (colored bars)
    • Raw speech regions (purple – threshold independent)
    • Overlay plot (green + purple)
    • Std probability histogram per segment
    • Threshold sweep comparison table
    • JSON with all timestamps (samples + seconds)
    • High-res PNGs saved to your output folder
- Fully typed, clean, no global state
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from jet.audio.norm.norm_speech_loudness import normalize_speech_loudness
from rich.console import Console
from rich.table import Table

console = Console()

model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True,
    verbose=False,
)
get_speech_timestamps, _, read_audio, _, _ = utils

class SegmentStats(TypedDict):
    avg_prob: float
    min_prob: float
    max_prob: float
    std_prob: float
    pct_above_threshold: float

Unit = Literal['ms', 'seconds']

@dataclass
class SpeechSegment:
    num: int
    start: int | float  # milliseconds or seconds
    end: int | float    # milliseconds or seconds
    duration: int | float  # milliseconds or seconds
    stats: SegmentStats

    def to_dict(self, *, timing_unit: Unit = 'ms') -> dict:
        """Return a dict representation with timing fields converted to seconds (3 decimals)."""
        data = asdict(self)
        factor = 1000.0 if timing_unit == 'ms' else 1.0
        for field in ('start', 'end', 'duration'):
            if data[field] is not None:
                data[field] = round(data[field] / factor, 3)
        return data

class SpeechAnalyzer:
    def __init__(
        self,
        threshold: float = 0.3,
        raw_threshold: float = 0.10,  # new: for more granular raw segments
        neg_threshold: float = 0.04,  # new: for more granular raw segments
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        # Increased speech_pad_ms for clear rise → peak → fall pattern on each segment
        speech_pad_ms: int = 0,
        sampling_rate: int = 16000,
        min_duration_ms: int | None = None,   # minimum raw segment duration in milliseconds
        min_std_prob: float | None = None,
        min_pct_threshold: float | None = None,
    ):
        self.threshold = threshold
        self.raw_threshold = raw_threshold
        self.neg_threshold = neg_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sr = sampling_rate
        self.min_duration_ms = min_duration_ms
        self.min_std_prob = min_std_prob
        self.min_pct_threshold = min_pct_threshold
        self.window_size_samples = 512 if sampling_rate == 16000 else 256
        self.frame_duration_ms = int(round((self.window_size_samples / self.sr) * 1000))  # 32 ms for both 8k/16k
        self.step_sec = self.window_size_samples / self.sr

    def extract_probs(self, wav: torch.Tensor) -> List[float]:
        """Extract raw speech probabilities for the entire audio waveform."""
        probs = []
        for i in range(0, len(wav), self.window_size_samples):
            chunk = wav[i : i + self.window_size_samples]
            if len(chunk) < self.window_size_samples:
                chunk = torch.nn.functional.pad(chunk, (0, self.window_size_samples - len(chunk)))
            prob = model(chunk.unsqueeze(0), self.sr).item()
            probs.append(prob)
        return probs

    def extract_energies(self, wav: torch.Tensor) -> List[float]:
        """
        Extract frame-level RMS energy aligned with VAD probability frames.
        Energy is normalized to [0, 1] for visualization & JSON export.
        """
        energies: List[float] = []
        for i in range(0, len(wav), self.window_size_samples):
            chunk = wav[i : i + self.window_size_samples]
            if len(chunk) < self.window_size_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, self.window_size_samples - len(chunk))
                )
            rms = torch.sqrt(torch.mean(chunk ** 2)).item()
            energies.append(rms)

        # Normalize safely
        max_val = max(energies) if energies else 1.0
        if max_val > 0:
            energies = [e / max_val for e in energies]

        return energies

    def extract_segments(
        self,
        segments: List[Dict[str, int]],
        prob_array: np.ndarray,
    ) -> List[SpeechSegment]:
        """Convert Silero's raw timestamp dicts into rich SpeechSegment objects."""
        rich_segments = []
        for idx, s in enumerate(segments):
            start_idx = int(s["start"] / self.window_size_samples)
            end_idx = int(s["end"] / self.window_size_samples)
            seg_probs = prob_array[start_idx:end_idx]
            start_ms = int(round(s["start"] / self.sr * 1000))
            end_ms = int(round(s["end"] / self.sr * 1000))
            duration_ms = end_ms - start_ms
            seg = SpeechSegment(
                num=idx + 1,
                start=start_ms,
                end=end_ms,
                duration=duration_ms,
                stats=SegmentStats(
                    avg_prob=round(float(seg_probs.mean()), 3) if len(seg_probs) > 0 else 0.0,
                    min_prob=round(float(seg_probs.min()), 3) if len(seg_probs) > 0 else 0.0,
                    max_prob=round(float(seg_probs.max()), 3) if len(seg_probs) > 0 else 0.0,
                    std_prob=round(float(seg_probs.std()), 3) if len(seg_probs) > 0 else 0.0,
                    pct_above_threshold=round(
                        sum(p > self.threshold for p in seg_probs) / len(seg_probs) * 100, 1
                    ) if len(seg_probs) > 0 else 0.0,
                ),
            )
            rich_segments.append(seg)
        return rich_segments

    def _segment_passes_filters(self, seg: SpeechSegment) -> bool:
        """Centralized filter logic – reusable across extraction and saving."""
        if self.min_duration_ms is not None and seg.duration < self.min_duration_ms:
            return False
        if self.min_std_prob is not None and seg.stats["std_prob"] < self.min_std_prob:
            return False
        if self.min_pct_threshold is not None and seg.stats["pct_above_threshold"] < self.min_pct_threshold:
            return False
        return True

    def extract_raw_segments(self, prob_array: np.ndarray) -> List[SpeechSegment]:
        """Create natural raw speech segments with low-probability boundaries."""
        raw_segments = []
        # Step 1: Find core speech regions using a higher threshold
        core_threshold = max(self.threshold, 0.6)  # at least 0.6, or use main threshold if higher
        is_core = prob_array > core_threshold

        # Step 2: Group contiguous core regions
        for group_key, group in itertools.groupby(enumerate(is_core), key=lambda x: x[1]):
            if not group_key:
                continue
            indices = [i for i, _ in group]
            if not indices:
                continue
            core_start = indices[0]
            core_end = indices[-1] + 1  # exclusive

            # Step 3: Expand backward until low prob or silence (use neg_threshold)
            expand_start = core_start
            while expand_start > 0 and prob_array[expand_start - 1] > self.neg_threshold:
                expand_start -= 1

            # Step 4: Expand forward (use neg_threshold)
            expand_end = core_end
            max_frames = len(prob_array)
            while expand_end < max_frames and prob_array[expand_end] > self.neg_threshold:
                expand_end += 1

            # Now the segment starts/ends near or in low-probability zones
            raw_probs = prob_array[expand_start:expand_end]
            if len(raw_probs) == 0:
                continue

            start_ms = int(round(expand_start * self.step_sec * 1000))
            end_ms = int(round(expand_end * self.step_sec * 1000))
            duration_ms = end_ms - start_ms

            raw_seg = SpeechSegment(
                num=len(raw_segments) + 1,
                start=start_ms,
                end=end_ms,
                duration=duration_ms,
                stats=SegmentStats(
                    avg_prob=round(float(raw_probs.mean()), 3),
                    min_prob=round(float(raw_probs.min()), 3),
                    max_prob=round(float(raw_probs.max()), 3),
                    std_prob=round(float(raw_probs.std()), 3),
                    pct_above_threshold=round(
                        float(np.sum(raw_probs > self.threshold)) / len(raw_probs) * 100, 1
                    ) if len(raw_probs) > 0 else 0.0,
                ),
            )

            # Optional: richer stats (helps debugging & tuning)
            raw_seg.stats["pct_above_raw"] = round(float(np.sum(raw_probs > self.raw_threshold)) / len(raw_probs) * 100, 1) if len(raw_probs) > 0 else 0.0
            raw_seg.stats["pct_above_neg"] = round(float(np.sum(raw_probs > self.neg_threshold)) / len(raw_probs) * 100, 1) if len(raw_probs) > 0 else 0.0

            if self._segment_passes_filters(raw_seg):
                raw_segments.append(raw_seg)

        return raw_segments

    def analyze(
        self,
        audio_path: str | Path,
        normalize: bool = True,           # ← new optional flag
    ) -> Tuple[List[float], List[float], List[SpeechSegment], List[SpeechSegment], int]:
        wav = read_audio(str(audio_path), sampling_rate=self.sr)

        if normalize:
            # Apply speech-aware loudness normalization as preprocessing
            wav = normalize_speech_loudness(
                wav.numpy(),                    # expects numpy array
                sample_rate=self.sr,
                target_lufs=-14.0,              # common podcast/YouTube speech target
                # You can tune the parameters below if needed:
                # min_lufs_threshold=-60.0,
                # max_loudness_threshold=-8.0,
                # peak_target=0.98,
                return_dtype=np.float32
            )
            wav = torch.from_numpy(wav).float()
        else:
            wav = wav.float()

        # Threshold-based segments from Silero
        segments = get_speech_timestamps(
            wav,
            model,
            threshold=self.threshold,
            sampling_rate=self.sr,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,
            visualize_probs=False,
        )
        model.reset_states()

        # Extract probabilities
        probs = self.extract_probs(wav)
        energies = self.extract_energies(wav)
        self._last_energies = energies
        prob_array = np.array(probs)
        num_frames = len(probs)

        # Rich threshold-based segments
        rich_segments = self.extract_segments(segments, prob_array)

        # Raw segments (independent of Silero's merging logic)
        raw_segments = self.extract_raw_segments(prob_array)

        return probs, energies, rich_segments, raw_segments, num_frames

    def plot_insights(
        self,
        probs: List[float],
        segments: List[SpeechSegment],
        raw_segments: List[SpeechSegment],
        num_frames: int,
        audio_path: str | Path,
        out_dir: str | Path,
    ) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        time_axis = [i * self.step_sec for i in range(num_frames)]
        total_sec = num_frames * self.step_sec

        # --- Energy timeline ---------------------------------------
        energies = getattr(self, "_last_energies", None)
        if energies is not None:
            fig, ax1 = plt.subplots(figsize=(20, 5))

            ax1.plot(time_axis, probs, color="steelblue", linewidth=1.5, label="Speech Probability")
            ax1.axhline(self.threshold, color="red", linestyle="--", alpha=0.7)
            ax1.set_ylabel("Speech Probability")
            ax1.set_ylim(0, 1.05)

            ax2 = ax1.twinx()
            ax2.plot(time_axis, energies, color="orange", alpha=0.7, linewidth=1.2, label="Energy")
            ax2.set_ylabel("Normalized Energy")
            ax2.set_ylim(0, 1.05)

            def set_dynamic_xticks(ax, total_seconds: float) -> None:
                target_ticks = 12
                interval = max(1, round(total_seconds / target_ticks, -1))
                if interval <= 5: interval = 5
                elif interval <= 12: interval = 10
                elif interval <= 35: interval = 30
                elif interval <= 70: interval = 60
                elif interval <= 150: interval = 120
                else: interval = 300
                ticks = np.arange(0, total_seconds + interval / 2, interval)
                labels = []
                for t in ticks:
                    if t >= 3600:
                        labels.append(f"{int(t // 3600)}:{int((t % 3600) // 60):02d}:{int(t % 60):02d}")
                    elif t >= 60:
                        labels.append(f"{int(t // 60):02d}:{int(t % 60):02d}")
                    else:
                        labels.append(f"{int(t)}s")
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels)
                ax.grid(True, which="major", axis="x", alpha=0.4, linestyle="--")
            set_dynamic_xticks(ax1, total_sec)
            ax1.set_xlabel("Time (seconds)")
            ax1.set_title(f"Probability + Energy Overlay – {Path(audio_path).name}")

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper right")

            overlay_path = Path(out_dir) / f"vad_prob_energy_overlay_{Path(audio_path).stem}.png"
            plt.savefig(overlay_path, dpi=300, bbox_inches="tight")
            print(f"Probability+Energy overlay saved → {overlay_path}")
            plt.close()

        def set_dynamic_xticks(ax, total_seconds: float) -> None:
            target_ticks = 12
            interval = max(1, round(total_seconds / target_ticks, -1))
            if interval <= 5: interval = 5
            elif interval <= 12: interval = 10
            elif interval <= 35: interval = 30
            elif interval <= 70: interval = 60
            elif interval <= 150: interval = 120
            else: interval = 300
            ticks = np.arange(0, total_seconds + interval / 2, interval)
            labels = []
            for t in ticks:
                if t >= 3600:
                    labels.append(f"{int(t // 3600)}:{int((t % 3600) // 60):02d}:{int(t % 60):02d}")
                elif t >= 60:
                    labels.append(f"{int(t // 60):02d}:{int(t % 60):02d}")
                else:
                    labels.append(f"{int(t)}s")
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.grid(True, which="major", axis="x", alpha=0.4, linestyle="--")

        # 1. Overlay plot (green + purple)
        plt.figure(figsize=(20, 6))
        ax = plt.gca()
        for i, seg in enumerate(raw_segments):
            ax.fill_between(
                [seg.start / 1000, seg.end / 1000], 0, 1,
                color="purple", alpha=0.4,
                label="Raw (>0.0)" if i == 0 else None
            )
        for i, seg in enumerate(segments):
            ax.fill_between(
                [seg.start / 1000, seg.end / 1000], 0, 1,
                color="lightgreen", edgecolor="black", linewidth=1, alpha=0.8,
                label="Detected Speech" if i == 0 else None
            )
            mid = (seg.start + seg.end) / 2000  # convert ms to seconds
            duration_sec = seg.duration / 1000
            ax.text(mid, 0.5, f"{duration_sec:.1f}s", ha="center", va="center",
                    color="darkgreen", fontweight="bold", fontsize=10)
        set_dynamic_xticks(ax, total_sec)
        plt.xlim(0, total_sec)
        plt.ylim(0, 1)
        plt.xlabel("Time (seconds)")
        plt.title(f"Overlay: Raw vs Detected Speech Segments – {Path(audio_path).name}")
        plt.legend()
        overlay_path = Path(out_dir) / f"vad_overlay_raw_vs_detected_{Path(audio_path).stem}.png"
        plt.savefig(overlay_path, dpi=300, bbox_inches="tight")
        print(f"Overlay plot saved → {overlay_path}")
        plt.close()

        # 2. Std probability histogram per segment
        if segments:
            std_values = [seg.stats["std_prob"] for seg in segments]
            plt.figure(figsize=(10, 6))
            plt.hist(std_values, bins=20, color="teal", alpha=0.7, edgecolor="black")
            plt.axvline(float(np.mean(std_values)), color="red", linestyle="--", label=f"Mean = {float(np.mean(std_values)):.3f}")
            plt.xlabel("Standard Deviation of Probability")
            plt.ylabel("Number of Segments")
            plt.title("Distribution of Probability Stability per Segment")
            plt.legend()
            std_hist_path = Path(out_dir) / f"vad_std_probability_histogram_{Path(audio_path).stem}.png"
            plt.savefig(std_hist_path, dpi=300, bbox_inches="tight")
            print(f"Std histogram saved → {std_hist_path}")
            plt.close()

        # Keep your original plots (they still work perfectly)
        plt.figure(figsize=(20, 10))
        plt.suptitle(f"Silero VAD Full Analysis – {Path(audio_path).name}", fontsize=18, fontweight="bold")
        plt.subplot(2, 1, 1)
        plt.fill_between(time_axis, probs, color="skyblue")
        plt.axhline(self.threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {self.threshold}")
        plt.ylim(0, 1.05)
        plt.xlim(0, total_sec)
        plt.ylabel("Probability")
        plt.title("Raw Speech Probability Over Time")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        set_dynamic_xticks(plt.gca(), total_sec)

        ax2 = plt.subplot(2, 1, 2)
        for i, seg in enumerate(segments):
            ax2.fill_between([seg.start / 1000, seg.end / 1000], 0, 1, color="lightgreen", edgecolor="black", linewidth=1, alpha=0.8,
                             label="Speech" if i == 0 else None)
            mid = (seg.start + seg.end) / 2000
            duration_sec = seg.duration / 1000
            ax2.text(mid, 0.5, f"{duration_sec:.1f}s", ha="center", va="center", color="darkgreen", fontweight="bold", fontsize=10)
        set_dynamic_xticks(ax2, total_sec)
        plt.xlim(0, total_sec)
        plt.ylim(0, 1)
        plt.xlabel("Time (seconds)")
        plt.title(f"Detected Speech Segments (n={len(segments)})")
        if segments:
            plt.legend()
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.subplots_adjust(hspace=0.3)
        plot_path = Path(out_dir) / f"vad_detected_speech_{Path(audio_path).stem}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved → {plot_path}")
        plt.close()

        # Histogram (speech vs silence)
        plt.figure(figsize=(12, 8))
        prob_array = np.array(probs)
        speech_probs = []
        for seg in segments:
            start_idx = int(seg.start / 1000 / self.step_sec)
            end_idx = int(seg.end / 1000 / self.step_sec)
            speech_probs.extend(prob_array[start_idx:end_idx])
        speech_probs = np.array(speech_probs)
        silence_mask = np.ones(len(prob_array), dtype=bool)
        for seg in segments:
            start_idx = int(seg.start / 1000 / self.step_sec)
            end_idx = int(seg.end / 1000 / self.step_sec)
            silence_mask[start_idx:end_idx] = False
        silence_probs = prob_array[silence_mask]
        plt.hist(silence_probs, bins=50, alpha=0.6, label="Silence", color="lightgray", density=True)
        plt.hist(speech_probs, bins=50, alpha=0.7, label="Speech", color="lightgreen", density=True)
        plt.axvline(self.threshold, color="red", linestyle="--", label=f"Threshold = {self.threshold}")
        plt.xlabel("Speech Probability")
        plt.ylabel("Density")
        plt.title("Probability Distribution (Speech vs. Silence)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        hist_path = Path(out_dir) / f"vad_probability_histogram_{Path(audio_path).stem}.png"
        plt.savefig(hist_path, dpi=300, bbox_inches="tight")
        print(f"Histogram saved → {hist_path}")
        plt.close()

        # Per-segment confidence
        if segments:
            plt.figure(figsize=(12, max(6, len(segments) * 0.4)))
            avg_probs = [seg.stats["avg_prob"] for seg in segments]
            durations = [seg.duration / 1000 for seg in segments]
            indices = np.arange(len(segments))
            colors = ["green" if p > 0.8 else "orange" if p > 0.6 else "red" for p in avg_probs]
            bars = plt.barh(indices, avg_probs, color=colors, alpha=0.8)
            plt.yticks(indices, [f"Seg {seg.num} ({d:.1f}s)" for seg, d in zip(segments, durations)])
            plt.xlabel("Average Probability")
            plt.title("Per-Segment Confidence (Avg Probability)")
            plt.xlim(0, 1)
            plt.xticks(np.linspace(0, 1, 11))
            for bar, prob in zip(bars, avg_probs):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{prob:.3f}",
                         va="center", fontsize=9, color="black", fontweight="bold")
            plt.grid(True, alpha=0.3, axis="x")
            conf_path = Path(out_dir) / f"vad_segment_confidence_{Path(audio_path).stem}.png"
            plt.savefig(conf_path, dpi=300, bbox_inches="tight")
            print(f"Segment confidence chart saved → {conf_path}")
            plt.close()

        # NEW: Per-segment probability timeline charts
        if segments:
            max_cols = 3
            n_segments = len(segments)
            cols = min(max_cols, n_segments)
            rows = (n_segments + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
            fig.suptitle(f"Per-Segment Probability Timeline – {Path(audio_path).name}", fontsize=16)

            prob_array = np.array(probs)
            for idx, seg in enumerate(segments):
                ax = axes[idx // cols, idx % cols]
                # Frame indices for this segment
                start_idx = int(round(seg.start / self.frame_duration_ms))
                end_idx = int(round(seg.end / self.frame_duration_ms))
                seg_probs = prob_array[start_idx:end_idx]
                # Relative time within segment (seconds)
                seg_time = np.arange(len(seg_probs)) * (self.frame_duration_ms / 1000.0)
                ax.plot(seg_time, seg_probs, color="steelblue", linewidth=1.5)
                ax.axhline(self.threshold, color="red", linestyle="--", alpha=0.7,
                           label=f"Threshold = {self.threshold}")
                ax.set_ylim(0, 1.05)
                ax.set_title(f"Seg {seg.num} ({seg.duration}ms | avg={seg.stats['avg_prob']:.3f})")
                ax.set_xlabel("Time in segment (s)")
                ax.set_ylabel("Probability")
                ax.grid(True, alpha=0.3)
                if idx == 0:
                    ax.legend(loc="upper right")

            # Hide unused subplots
            for idx in range(n_segments, rows * cols):
                axes[idx // cols, idx % cols].set_visible(False)

            plt.tight_layout(rect=(0, 0, 1, 0.96))
            per_seg_path = Path(out_dir) / f"vad_per_segment_probability_{Path(audio_path).stem}.png"
            plt.savefig(per_seg_path, dpi=300, bbox_inches="tight")
            print(f"Per-segment probability timeline saved → {per_seg_path}")
            plt.close()

    def save_json(
        self,
        segments: List[SpeechSegment],
        out_dir: str | Path,
        audio_path: str | Path,
        *,
        extra_info: dict | None = None,
    ) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        data = {
            "audio_file": Path(audio_path).name,
            "threshold": self.threshold,
            "sampling_rate": self.sr,
            "frame_duration_ms": self.frame_duration_ms,
            "total_frames": extra_info.get("total_frames") if extra_info else None,
            "total_segments": len(segments),
            "segments": [seg.to_dict(timing_unit="ms") for seg in segments],
        }
        if extra_info:
            data.update(extra_info)
        json_path = Path(out_dir) / f"vad_segments_{Path(audio_path).stem}.json"
        json_path.write_text(json.dumps(data, indent=2))
        print(f"JSON saved → {json_path}")

    def save_raw_json(
        self,
        raw_segments: List[SpeechSegment],
        out_dir: str | Path,
        audio_path: str | Path,
        *,
        extra_info: dict | None = None,
    ) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        data = {
            "audio_file": Path(audio_path).name,
            "note": f"Raw contiguous regions with probability > {self.raw_threshold} (tunable raw threshold)",
            "sampling_rate": self.sr,
            "raw_threshold": self.raw_threshold,
            "frame_duration_ms": self.frame_duration_ms,
            "total_frames": extra_info.get("total_frames") if extra_info else None,
            "total_raw_segments": len(raw_segments),
            "segments": [seg.to_dict(timing_unit="ms") for seg in raw_segments],
        }
        if extra_info:
            data.update(extra_info)
        json_path = Path(out_dir) / f"vad_raw_segments_{Path(audio_path).stem}.json"
        json_path.write_text(json.dumps(data, indent=2))
        print(f"Raw JSON saved → {json_path}")

    def save_energies_json(
        self,
        energies: List[float],
        out_dir: str | Path,
        audio_path: str | Path,
    ) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        data = {
            "audio_file": Path(audio_path).name,
            "frame_duration_ms": self.frame_duration_ms,
            "sampling_rate": self.sr,
            "num_frames": len(energies),
            "energies": [round(float(e), 6) for e in energies],
        }
        path = Path(out_dir) / f"vad_energies_{Path(audio_path).stem}.json"
        path.write_text(json.dumps(data, indent=2))
        print(f"Energies JSON saved → {path}")

    def _save_segments_individually(
        self,
        audio_path: str | Path,
        segments: List[SpeechSegment],
        out_dir: str | Path,
        probs: List[float],
        *,
        prefix: str = "segment",           # "segment" or "raw_segment"
        subdir: str | None = None,         # optional subdir like "raw_segments"
        apply_filters: bool = False,        # whether to skip filtered segments
        segment_type: str | None = None,   # added to meta if provided
        chart_color: str = "steelblue",    # line color
        show_raw_threshold: bool = False,  # draw raw_threshold line
        title_prefix: str = "",            # e.g. "Raw "
    ) -> int:
        """
        Generic private method to save individual segments (detected or raw).

        Returns:
            int: number of segments actually saved
        """
        base_dir = Path(out_dir)
        if subdir:
            base_dir = base_dir / subdir
        base_dir.mkdir(parents=True, exist_ok=True)

        wav, sr = sf.read(str(audio_path))
        if sr != self.sr:
            raise ValueError(f"Audio sampling rate {sr} does not match analyzer's {self.sr}")

        prob_array = np.array(probs)
        saved_count = 0

        for idx, seg in enumerate(segments, start=1):
            if apply_filters and not self._segment_passes_filters(seg):
                continue

            seg_dir = base_dir / f"{prefix}_{seg.num:03d}"
            seg_dir.mkdir(parents=True, exist_ok=True)

            start_sample = int(seg.start / 1000 * self.sr)
            end_sample = int(seg.end / 1000 * self.sr)
            segment_audio = wav[start_sample:end_sample]
            sf.write(str(seg_dir / "sound.wav"), segment_audio, self.sr)

            meta = seg.to_dict()
            meta.update({
                "segment_index": idx,
                "original_file": Path(audio_path).name,
                "frame_duration_ms": self.frame_duration_ms,
                "segment_frame_count": int(round(seg.duration / self.frame_duration_ms)),
            })

            # Add segment energy stats if available
            energies = np.array(getattr(self, "_last_energies", []))
            start_idx = int(round(seg.start / self.frame_duration_ms))
            end_idx = int(round(seg.end / self.frame_duration_ms))
            seg_energies = energies[start_idx:end_idx] if energies.size else None

            if seg_energies is not None and len(seg_energies):
                meta["energy_stats"] = {
                    "avg_energy": round(float(seg_energies.mean()), 5),
                    "peak_energy": round(float(seg_energies.max()), 5),
                    "energy_std": round(float(seg_energies.std()), 5),
                }

            if segment_type:
                meta["segment_type"] = segment_type
            if apply_filters:
                meta["applied_filters"] = {
                    "min_duration_ms": self.min_duration_ms,
                    "min_std_prob": self.min_std_prob,
                    "min_pct_threshold": self.min_pct_threshold,
                }
            (seg_dir / "meta.json").write_text(json.dumps(meta, indent=2))

            # Per-segment probability chart
            seg_probs = prob_array[start_idx:end_idx]
            if len(seg_probs) == 0:
                continue  # safety
            seg_time_sec = np.arange(len(seg_probs)) * (self.frame_duration_ms / 1000.0)

            plt.figure(figsize=(8, 4))
            plt.plot(seg_time_sec, seg_probs, color=chart_color, linewidth=1.8, label="Probability")

            if seg_energies is not None and len(seg_energies) == len(seg_probs):
                plt.plot(seg_time_sec, seg_energies, color="orange", alpha=0.6,
                         linewidth=1.2, label="Energy")

            plt.axhline(self.threshold, color="red", linestyle="--", alpha=0.8,
                        label=f"Threshold = {self.threshold}")
            if show_raw_threshold:
                plt.axhline(self.raw_threshold, color="orange", linestyle=":", alpha=0.7,
                            label=f"Raw threshold = {self.raw_threshold}")
            plt.ylim(0, 1.05)
            plt.title(f"{title_prefix}Segment {seg.num} Probability Timeline\n"
                      f"Duration: {seg.duration}ms | Avg: {seg.stats['avg_prob']:.3f} | "
                      f"Frames: {len(seg_probs)}")
            plt.xlabel("Time in segment (seconds)")
            plt.ylabel("Speech Probability")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="upper right")
            plt.tight_layout()

            chart_path = seg_dir / "probability_timeline.png"
            plt.savefig(chart_path, dpi=300)
            plt.close()
            print(f"  → {title_prefix}segment chart saved: {chart_path}")

            saved_count += 1

        total = len(segments)
        if apply_filters:
            print(f"Saved {saved_count}/{total} filtered {title_prefix.lower()}segments → {base_dir}")
        else:
            print(f"Saved {saved_count} {title_prefix.lower()}segments → {base_dir}")
        return saved_count

    def save_segments_individually(
        self,
        audio_path: str | Path,
        segments: List[SpeechSegment],
        out_dir: str | Path,
        probs: List[float],
    ) -> None:
        self._save_segments_individually(
            audio_path=audio_path,
            segments=segments,
            out_dir=out_dir,
            probs=probs,
            prefix="segment",
            subdir=None,
            apply_filters=False,
            chart_color="steelblue",
            title_prefix="",
        )

    def save_raw_segments_individually(
        self,
        audio_path: str | Path,
        raw_segments: List[SpeechSegment],
        out_dir: str | Path,
        probs: List[float],
    ) -> None:
        self._save_segments_individually(
            audio_path=audio_path,
            segments=raw_segments,
            out_dir=out_dir,
            probs=probs,
            prefix="raw_segment",
            subdir="raw_segments",
            apply_filters=True,
            segment_type="raw",
            chart_color="purple",
            show_raw_threshold=True,
            title_prefix="Raw ",
        )

    def get_metrics(
        self,
        probs: List[float],
        segments: List[SpeechSegment],
        raw_segments: List[SpeechSegment],
        num_frames: int,
        total_duration_sec: float,
    ) -> dict:
        durations = [s.duration / 1000 for s in segments]
        num_segments = len(segments)
        total_speech_sec = sum(durations)
        speech_percent = total_speech_sec / total_duration_sec * 100 if total_duration_sec else 0

        raw_durations = [s.duration / 1000 for s in raw_segments]
        raw_total_speech_sec = sum(raw_durations)
        raw_speech_percent = raw_total_speech_sec / total_duration_sec * 100 if total_duration_sec else 0

        prob_array = np.array(probs)

        energies = np.array(getattr(self, "_last_energies", []))

        avg_energy = float(energies.mean()) if energies.size else 0.0
        peak_energy = float(energies.max()) if energies.size else 0.0
        energy_std = float(energies.std()) if energies.size else 0.0

        # Mask for frames classified as speech (threshold-based segments)
        speech_mask = np.zeros(len(probs), dtype=bool)
        for seg in segments:
            start_idx = int(seg.start / 1000 / self.step_sec)
            end_idx = int(seg.end / 1000 / self.step_sec)
            speech_mask[start_idx:end_idx] = True

        avg_prob_speech = float(prob_array[speech_mask].mean()) if speech_mask.any() else 0.0
        avg_prob_silence = float(prob_array[~speech_mask].mean()) if (~speech_mask).any() else 0.0

        # Average standard deviation per detected segment (measure of probability fluctuation)
        seg_stds = [seg.stats["std_prob"] for seg in segments]
        avg_std_per_segment = float(np.mean(seg_stds)) if seg_stds else 0.0

        # Fragmentation score: how well detected segments align with raw natural regions
        raw_num = len(raw_segments)
        fragmentation_score = round(num_segments / raw_num, 3) if raw_num > 0 else 0.0

        # Gaps between consecutive detected segments
        gaps = [
            (segments[i + 1].start - segments[i].end) / 1000
            for i in range(num_segments - 1)
        ] if num_segments > 1 else []
        avg_gap_sec = float(np.mean(gaps)) if gaps else 0.0

        metrics = {
            "total_duration_sec": round(total_duration_sec, 3),
            "total_speech_sec": round(total_speech_sec, 3),
            "speech_percentage": round(speech_percent, 1),
            "total_silence_sec": round(total_duration_sec - total_speech_sec, 3),
            "num_segments": num_segments,
            "avg_segment_sec": round(float(np.mean(durations)), 3) if durations else 0,
            "raw_num_segments": len(raw_segments),
            "raw_total_speech_sec": round(raw_total_speech_sec, 3),
            "raw_speech_percentage": round(raw_speech_percent, 1),
            "raw_avg_segment_sec": round(float(np.mean(raw_durations)), 3) if raw_durations else 0,
            "avg_gap_between_segments_sec": round(avg_gap_sec, 3),
            "avg_probability_all": round(float(prob_array.mean()), 3),
            "avg_probability_in_speech": round(avg_prob_speech, 3),
            "avg_probability_in_silence": round(avg_prob_silence, 3),
            "windows_above_threshold_percent": round(
                sum(p > self.threshold for p in probs) / len(probs) * 100, 1
            ),
            # Insightful metrics for threshold sweep
            "avg_std_per_segment": round(avg_std_per_segment, 3),
            "fragmentation_score": fragmentation_score,
            # Frame-level metadata
            "frame_duration_ms": self.frame_duration_ms,
            "total_frames": num_frames,
            "frames_per_second": round(1000.0 / self.frame_duration_ms, 1),
        }

        if energies.size > 0:
            metrics.update({
                "energy_min": round(float(np.min(energies)), 4) if energies.size else 0,
                "energy_max": round(float(np.max(energies)), 4) if energies.size else 0,
                "energy_mean": round(float(np.mean(energies)), 4) if energies.size else 0,
                "energy_std": round(float(np.std(energies)), 4) if energies.size else 0,
            })
        else:
            metrics.update({
                "energy_min": 0,
                "energy_max": 0,
                "energy_mean": 0,
                "energy_std": 0,
            })

        return metrics

    def run_threshold_sweep(
        self,
        audio_path: str | Path,
        thresholds: List[float] = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9],
    ) -> List[Dict]:
        results = []
        wav = read_audio(str(audio_path), sampling_rate=self.sr).float()
        probs = self.extract_probs(wav)
        total_sec = len(probs) * self.step_sec

        for t in thresholds:
            analyzer = SpeechAnalyzer(
                threshold=t,
                raw_threshold=self.raw_threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
                sampling_rate=self.sr,
                min_duration_ms=self.min_duration_ms,
                min_std_prob=self.min_std_prob,
                min_pct_threshold=self.min_pct_threshold,
            )
            _, _, segments, raw_segments, _ = analyzer.analyze(audio_path)
            metrics = analyzer.get_metrics(probs, segments, raw_segments, len(probs), total_sec)

            results.append({
                "threshold": t,
                "num_segments": metrics["num_segments"],
                "speech_percentage": metrics["speech_percentage"],
                "raw_num_segments": metrics["raw_num_segments"],
                "raw_speech_percentage": metrics["raw_speech_percentage"],
                "avg_segment_sec": metrics["avg_segment_sec"],
                "avg_prob_speech": metrics["avg_probability_in_speech"],
                "avg_std_per_segment": metrics["avg_std_per_segment"],
                "fragmentation_score": metrics["fragmentation_score"],
            })

        return results

    def save_threshold_sweep(self, sweep_results: List[Dict], out_dir: str | Path, audio_path: str | Path) -> None:
        path = Path(out_dir) / f"vad_threshold_sweep_{Path(audio_path).stem}.json"
        path.write_text(json.dumps(sweep_results, indent=2))
        print(f"Threshold sweep saved → {path}")

    def print_threshold_sweep_table(self, sweep_results: List[Dict]) -> None:
        # Calculate composite score with balanced weights
        for r in sweep_results:
            # Penalize extreme over-merging (segments > ~5s feel unnatural)
            over_merge_penalty = max(0, (r["avg_segment_sec"] - 5.0) * 1.5)
            # Reward higher confidence, lower fluctuation, and good alignment to raw regions
            # Also lightly favor moderate speech coverage (~40-60%)
            speech_coverage_bonus = 10 * (1 - abs(r["speech_percentage"] - 50) / 50)

            r["composite"] = (
                r["avg_prob_speech"] * 12.0          # strong weight on confidence
                + (1 - r["avg_std_per_segment"]) * 8.0  # stability
                + r["fragmentation_score"] * 6.0     # alignment to natural regions
                + speech_coverage_bonus              # prefer balanced recall
                - over_merge_penalty
            )

        # Rank by composite score
        ranked = sorted(sweep_results, key=lambda x: x["composite"], reverse=True)
        for rank, r in enumerate(ranked, 1):
            r["rank"] = rank

        table = Table(title="Threshold Sweep Comparison (Improved & Balanced)")
        table.add_column("Rank", justify="center", style="bold")
        table.add_column("Threshold", justify="right")
        table.add_column("Num Seg", justify="right")
        table.add_column("Speech %", justify="right")
        table.add_column("Raw Num", justify="right")
        table.add_column("Raw Speech %", justify="right")
        table.add_column("Avg Seg Sec", justify="right")
        table.add_column("Avg Prob Speech", justify="right")
        table.add_column("Avg Std/Seg", justify="right")
        table.add_column("Frag Score", justify="right")
        table.add_column("Composite", justify="right")

        for r in ranked:
            if r["rank"] == 1:
                rank_text = "[bold green]1[/bold green]"
            elif r["rank"] == 2:
                rank_text = "[bold yellow]2[/bold yellow]"
            else:
                rank_text = str(r["rank"])

            table.add_row(
                rank_text,
                f"{r['threshold']:.1f}",
                str(r["num_segments"]),
                f"{r['speech_percentage']}%",
                str(r["raw_num_segments"]),
                f"{r['raw_speech_percentage']}%",
                f"{r['avg_segment_sec']:.2f}s",
                f"{r['avg_prob_speech']:.3f}",
                f"{r['avg_std_per_segment']:.3f}",
                f"{r['fragmentation_score']:.2f}",
                f"{r['composite']:.1f}",
            )

        console.print(table)

def main():
    import shutil

    from jet.file.utils import save_file

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    DEFAULT_AUDIO_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/full_recording.wav"

    parser = argparse.ArgumentParser(description="Silero VAD Full Analyzer – Beautiful insights + JSON")
    parser.add_argument(
        "audio",
        type=Path,
        nargs="?",
        default=Path(DEFAULT_AUDIO_PATH),
        help=f"Path to input .wav file (default: {DEFAULT_AUDIO_PATH})"
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=Path(OUTPUT_DIR))
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("--raw-threshold", type=float, default=0.10,
                        help="Threshold for raw segments (default: 0.10, set 0.0 for original behavior)")
    parser.add_argument("--min-duration-ms", type=int, default=200, help="Minimum raw segment duration in ms (default: 200)")
    parser.add_argument("--min-std-prob", type=float, default=0.0, help="Minimum std(prob) in raw region (default: 0.0)")
    parser.add_argument("--min-pct-threshold", type=float, default=10.0, help="Min %% windows > threshold in raw region (default: 10.0)")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Audio not found: {args.audio}")
        sys.exit(1)

    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analyzer = SpeechAnalyzer(
        threshold=args.threshold,
        raw_threshold=args.raw_threshold,
        min_duration_ms=args.min_duration_ms,
        min_std_prob=args.min_std_prob,
        min_pct_threshold=args.min_pct_threshold,
    )
    print(f"Analyzing: {args.audio.name}")
    print(f"Threshold: {args.threshold} | Output → {args.output_dir.resolve()}")

    probs, energies, segments, raw_segments, num_frames = analyzer.analyze(args.audio)
    total_sec = num_frames * analyzer.step_sec
    metrics = analyzer.get_metrics(probs, segments, raw_segments, num_frames, total_sec)
    analyzer.plot_insights(probs, segments, raw_segments, num_frames, args.audio, args.output_dir)

    extra_info = {
        "total_frames": num_frames,
        "total_duration_sec": round(total_sec, 3),
        "frame_duration_ms": analyzer.frame_duration_ms,
    }
    analyzer.save_json(segments, args.output_dir, args.audio, extra_info=extra_info)
    analyzer.save_raw_json(raw_segments, args.output_dir, args.audio, extra_info=extra_info)
    analyzer.save_energies_json(energies, args.output_dir, args.audio)
    analyzer.save_segments_individually(
        args.audio,
        segments,
        args.output_dir / "segments",
        probs,  # Pass full probs for per-segment charts
    )
    analyzer.save_raw_segments_individually(
        args.audio,
        raw_segments,
        args.output_dir,
        probs,  # Pass full probs for per-segment charts
    )

    from rich.console import Console
    from rich.table import Table
    console = Console()
    table = Table(title=f"[bold]VAD Metrics – {args.audio.name}[/bold]")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in metrics.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)

    metrics_path = args.output_dir / f"vad_metrics_{Path(args.audio).stem}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics JSON → {metrics_path}")

    sweep = analyzer.run_threshold_sweep(args.audio)
    analyzer.save_threshold_sweep(sweep, args.output_dir, args.audio)
    analyzer.print_threshold_sweep_table(sweep)

    print("\nAll done! Open the PNGs + overlay + std histogram + sweep table.")
    print(f"→ {args.output_dir}")

    formatted_segments = [seg.to_dict() for seg in segments]
    formatted_raw_segments = [seg.to_dict() for seg in raw_segments]

    save_file(probs, f"{str(args.output_dir)}/probs.json")
    save_file(energies, f"{str(args.output_dir)}/energies.json")
    save_file(formatted_segments, f"{str(args.output_dir)}/segments.json")
    save_file(formatted_raw_segments, f"{str(args.output_dir)}/raw_segments.json")
    save_file(metrics, f"{str(args.output_dir)}/vad_metrics.json")

if __name__ == "__main__":
    main()