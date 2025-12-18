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
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Literal, Tuple, Dict
from typing import TypedDict
import numpy as np
import matplotlib.pyplot as plt
import torch
import soundfile as sf
import itertools

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
        threshold: float = 0.5,
        raw_threshold: float = 0.2,  # new: for more granular raw segments
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        sampling_rate: int = 16000,
        min_duration_ms: int | None = None,   # minimum raw segment duration in milliseconds
        min_std_prob: float | None = None,
        min_pct_threshold: float | None = None,
    ):
        self.threshold = threshold
        self.raw_threshold = raw_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sr = sampling_rate
        self.min_duration_ms = min_duration_ms
        self.min_std_prob = min_std_prob
        self.min_pct_threshold = min_pct_threshold
        self.window_size = 512 if sampling_rate == 16000 else 256
        self.step_sec = self.window_size / self.sr

    def extract_probs(self, wav: torch.Tensor) -> List[float]:
        """Extract raw speech probabilities for the entire audio waveform."""
        probs = []
        for i in range(0, len(wav), self.window_size):
            chunk = wav[i : i + self.window_size]
            if len(chunk) < self.window_size:
                chunk = torch.nn.functional.pad(chunk, (0, self.window_size - len(chunk)))
            prob = model(chunk.unsqueeze(0), self.sr).item()
            probs.append(prob)
        return probs

    def extract_segments(
        self,
        segments: List[Dict[str, int]],
        prob_array: np.ndarray,
    ) -> List[SpeechSegment]:
        """Convert Silero's raw timestamp dicts into rich SpeechSegment objects."""
        rich_segments = []
        for idx, s in enumerate(segments):
            start_idx = int(s["start"] / self.window_size)
            end_idx = int(s["end"] / self.window_size)
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
        """Create raw speech segments based on any probability > raw_threshold (contiguous regions)."""
        raw_segments = []
        is_speech = prob_array > self.raw_threshold
        for idx, (is_speech_key, group) in enumerate(itertools.groupby(enumerate(is_speech), key=lambda x: x[1])):
            if not is_speech_key:  # skip silence groups
                continue
            indices = [i for i, _ in group]
            start_idx = indices[0]
            end_idx = indices[-1] + 1
            raw_probs = prob_array[start_idx:end_idx]
            if len(raw_probs) == 0:
                continue
            start_ms = int(round(start_idx * self.step_sec * 1000))
            end_ms = int(round(end_idx * self.step_sec * 1000))
            duration_ms = end_ms - start_ms
            raw_seg = SpeechSegment(
                num=idx + 1,
                start=start_ms,
                end=end_ms,
                duration=duration_ms,
                stats=SegmentStats(
                    avg_prob=round(float(raw_probs.mean()), 3),
                    min_prob=round(float(raw_probs.min()), 3),
                    max_prob=round(float(raw_probs.max()), 3),
                    std_prob=round(float(raw_probs.std()), 3),
                    pct_above_threshold=round(
                        float(np.sum(raw_probs > self.threshold)) / float(len(raw_probs)) * 100, 1
                    ),
                ),
            )
            if self._segment_passes_filters(raw_seg):
                raw_segments.append(raw_seg)
        return raw_segments

    def analyze(self, audio_path: str | Path) -> Tuple[List[float], List[SpeechSegment], List[SpeechSegment]]:
        wav = read_audio(str(audio_path), sampling_rate=self.sr).float()

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
        prob_array = np.array(probs)

        # Rich threshold-based segments
        rich_segments = self.extract_segments(segments, prob_array)

        # Raw segments (independent of Silero's merging logic)
        raw_segments = self.extract_raw_segments(prob_array)

        return probs, rich_segments, raw_segments

    def plot_insights(
        self,
        probs: List[float],
        segments: List[SpeechSegment],
        raw_segments: List[SpeechSegment],
        audio_path: str | Path,
        out_dir: str | Path,
    ) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        time_axis = [i * self.step_sec for i in range(len(probs))]
        total_sec = len(probs) * self.step_sec

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

    def save_json(self, segments: List[SpeechSegment], out_dir: str | Path, audio_path: str | Path) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        data = {
            "audio_file": Path(audio_path).name,
            "threshold": self.threshold,
            "sampling_rate": self.sr,
            "total_segments": len(segments),
            "segments": [asdict(s) for s in segments],
        }
        json_path = Path(out_dir) / f"vad_segments_{Path(audio_path).stem}.json"
        json_path.write_text(json.dumps(data, indent=2))
        print(f"JSON saved → {json_path}")

    def save_raw_json(self, raw_segments: List[SpeechSegment], out_dir: str | Path, audio_path: str | Path) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        data = {
            "audio_file": Path(audio_path).name,
            "note": f"Raw contiguous regions with probability > {self.raw_threshold} (tunable raw threshold)",
            "sampling_rate": self.sr,
            "raw_threshold": self.raw_threshold,
            "total_raw_segments": len(raw_segments),
            "segments": [asdict(s) for s in raw_segments],
        }
        json_path = Path(out_dir) / f"vad_raw_segments_{Path(audio_path).stem}.json"
        json_path.write_text(json.dumps(data, indent=2))
        print(f"Raw JSON saved → {json_path}")

    def save_segments_individually(
        self,
        audio_path: str | Path,
        segments: List[SpeechSegment],
        out_dir: str | Path,
    ) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        wav, sr = sf.read(str(audio_path))
        if sr != self.sr:
            raise ValueError(f"Audio sampling rate {sr} does not match analyzer's {self.sr}")

        for idx, seg in enumerate(segments, start=1):
            seg_dir = Path(out_dir) / f"segment_{seg.num:03d}"
            seg_dir.mkdir(parents=True, exist_ok=True)

            start_sample = int(seg.start / 1000 * self.sr)
            end_sample = int(seg.end / 1000 * self.sr)
            segment_audio = wav[start_sample:end_sample]

            sf.write(str(seg_dir / "sound.wav"), segment_audio, self.sr)

            meta = seg.to_dict()
            meta.update({
                "segment_index": idx,
                "original_file": Path(audio_path).name,
            })
            (seg_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"Saved {len(segments)} individual segments → {out_dir}")

    def save_raw_segments_individually(
        self,
        audio_path: str | Path,
        raw_segments: List[SpeechSegment],
        out_dir: str | Path,
    ) -> None:
        """
        Save each raw segment as an individual subdirectory containing:
        - sound.wav (extracted audio chunk)
        - meta.json (segment metadata)

        Filters are applied based on analyzer-level settings:
        - self.min_duration_ms
        - self.min_std_prob
        - self.min_pct_threshold

        Directory structure: out_dir/raw_segments/raw_segment_001, ...
        """
        raw_dir = Path(out_dir) / "raw_segments"
        raw_dir.mkdir(parents=True, exist_ok=True)

        wav, sr = sf.read(str(audio_path))
        if sr != self.sr:
            raise ValueError(f"Audio sampling rate {sr} does not match analyzer's {self.sr}")

        saved_count = 0
        for idx, seg in enumerate(raw_segments, start=1):
            if not self._segment_passes_filters(seg):
                continue

            seg_dir = raw_dir / f"raw_segment_{seg.num:03d}"
            seg_dir.mkdir(parents=True, exist_ok=True)

            start_sample = int(seg.start / 1000 * self.sr)
            end_sample = int(seg.end / 1000 * self.sr)
            segment_audio = wav[start_sample:end_sample]

            sf.write(str(seg_dir / "sound.wav"), segment_audio, self.sr)

            meta = seg.to_dict()
            meta.update(
                {
                    "segment_index": idx,
                    "segment_type": "raw",
                    "original_file": Path(audio_path).name,
                    "applied_filters": {
                        "min_duration_ms": self.min_duration_ms,
                        "min_std_prob": self.min_std_prob,
                        "min_pct_threshold": self.min_pct_threshold,
                    },
                }
            )
            (seg_dir / "meta.json").write_text(json.dumps(meta, indent=2))

            saved_count += 1

        print(f"Saved {saved_count}/{len(raw_segments)} filtered raw segments → {raw_dir}")

    def get_metrics(
        self,
        probs: List[float],
        segments: List[SpeechSegment],
        raw_segments: List[SpeechSegment],
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
        speech_mask = np.zeros(len(probs), dtype=bool)
        for seg in segments:
            start_idx = int(seg.start / 1000 / self.step_sec)
            end_idx = int(seg.end / 1000 / self.step_sec)
            speech_mask[start_idx:end_idx] = True

        avg_prob_speech = float(prob_array[speech_mask].mean()) if speech_mask.any() else 0.0
        avg_prob_silence = float(prob_array[~speech_mask].mean()) if (~speech_mask).any() else 0.0

        gaps = [((segments[i + 1].start - segments[i].end) / 1000) for i in range(num_segments - 1)] if num_segments > 1 else []
        avg_gap_sec = float(np.mean(gaps)) if gaps else 0.0

        return {
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
            "windows_above_threshold_percent": round(sum(p > self.threshold for p in probs) / len(probs) * 100, 1),
        }

    def run_threshold_sweep(self, audio_path: str | Path, thresholds: List[float] = [0.3, 0.5, 0.7]) -> List[Dict]:
        results = []
        wav = read_audio(str(audio_path), sampling_rate=self.sr).float()
        probs = []
        for i in range(0, len(wav), self.window_size):
            chunk = wav[i : i + self.window_size]
            if len(chunk) < self.window_size:
                chunk = torch.nn.functional.pad(chunk, (0, self.window_size - len(chunk)))
            prob = model(chunk.unsqueeze(0), self.sr).item()
            probs.append(prob)
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
            _, segments, raw_segments = analyzer.analyze(audio_path)
            metrics = analyzer.get_metrics(probs, segments, raw_segments, total_sec)
            results.append({
                "threshold": t,
                "num_segments": metrics["num_segments"],
                "speech_percentage": metrics["speech_percentage"],
                "raw_num_segments": metrics["raw_num_segments"],
                "raw_speech_percentage": metrics["raw_speech_percentage"],
                "avg_segment_sec": metrics["avg_segment_sec"],
            })
        return results

    def save_threshold_sweep(self, sweep_results: List[Dict], out_dir: str | Path, audio_path: str | Path) -> None:
        path = Path(out_dir) / f"vad_threshold_sweep_{Path(audio_path).stem}.json"
        path.write_text(json.dumps(sweep_results, indent=2))
        print(f"Threshold sweep saved → {path}")

    def print_threshold_sweep_table(self, sweep_results: List[Dict]) -> None:
        from rich.table import Table
        from rich.console import Console
        console = Console()
        table = Table(title="Threshold Sweep Comparison")
        table.add_column("Threshold")
        table.add_column("Num Segments")
        table.add_column("Speech %")
        table.add_column("Raw Num")
        table.add_column("Raw Speech %")
        table.add_column("Avg Seg Sec")
        for r in sweep_results:
            table.add_row(
                str(r["threshold"]),
                str(r["num_segments"]),
                f"{r['speech_percentage']}%",
                str(r["raw_num_segments"]),
                f"{r['raw_speech_percentage']}%",
                f"{r['avg_segment_sec']:.2f}s",
            )
        console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Silero VAD Full Analyzer – Beautiful insights + JSON")
    parser.add_argument("audio", type=Path, help="Path to input .wav file")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("vad_results"))
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("--raw-threshold", type=float, default=0.2,
                        help="Threshold for raw segments (default: 0.2, set 0.0 for original behavior)")
    parser.add_argument("--min-duration-ms", type=int, default=200, help="Minimum raw segment duration in ms (default: 200)")
    parser.add_argument("--min-std-prob", type=float, default=0.0, help="Minimum std(prob) in raw region (default: 0.0)")
    parser.add_argument("--min-pct-threshold", type=float, default=10.0, help="Min %% windows > threshold in raw region (default: 10.0)")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Audio not found: {args.audio}")
        sys.exit(1)

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

    probs, segments, raw_segments = analyzer.analyze(args.audio)
    total_sec = len(probs) * analyzer.step_sec
    metrics = analyzer.get_metrics(probs, segments, raw_segments, total_sec)
    analyzer.plot_insights(probs, segments, raw_segments, args.audio, args.output_dir)
    analyzer.save_json(segments, args.output_dir, args.audio)
    analyzer.save_raw_json(raw_segments, args.output_dir, args.audio)
    analyzer.save_segments_individually(args.audio, segments, args.output_dir / "segments")
    analyzer.save_raw_segments_individually(
        args.audio,
        raw_segments,
        args.output_dir,
    )

    from rich.table import Table
    from rich.console import Console
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

if __name__ == "__main__":
    main()