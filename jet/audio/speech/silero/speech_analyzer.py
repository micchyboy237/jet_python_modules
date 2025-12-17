# file: speech_analyzer.py
"""
Silero VAD Analyzer – One-click full insight generator

Features:
- Loads any audio file
- Runs Silero VAD with your chosen threshold
- Generates:
    • Interactive probability plot (with threshold lines)
    • Speech segments timeline (colored bars)
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
from typing import List, Tuple
import numpy as np

import matplotlib.pyplot as plt
import torch
import soundfile as sf  # NEW: for audio saving

# ----------------------------------------------------------------------
# Load Silero VAD (official way – always up to date)
# ----------------------------------------------------------------------
model, utils = torch.hub.load(  # pyright: ignore[reportGeneralTypeIssues]
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True,
    verbose=False,
)

get_speech_timestamps, _, read_audio, _, _ = utils


@dataclass
class SpeechSegment:
    start_sec: float
    end_sec: float
    duration_sec: float
    avg_probability: float = 0.0
    min_probability: float = 0.0
    max_probability: float = 0.0
    std_probability: float = 0.0
    percent_above_threshold: float = 0.0


class SileroVADAnalyzer:
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        sampling_rate: int = 16000,
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sr = sampling_rate

        self.window_size = 512 if sampling_rate == 16000 else 256
        self.step_sec = self.window_size / self.sr  # 0.032s or 0.032s

    def analyze(self, audio_path: str | Path) -> Tuple[List[float], List[SpeechSegment]]:
        wav = read_audio(str(audio_path), sampling_rate=self.sr)

        # 1. Get speech timestamps (with internal probability collection)
        segments = get_speech_timestamps(
            wav,
            model,
            threshold=self.threshold,
            sampling_rate=self.sr,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,  # we'll convert ourselves
            visualize_probs=False,  # we do it manually for beauty
        )

        # 2. Manually collect probabilities (for full control + plotting)
        model.reset_states()
        probs = []
        for i in range(0, len(wav), self.window_size):
            chunk = wav[i : i + self.window_size]
            if len(chunk) < self.window_size:
                chunk = torch.nn.functional.pad(chunk, (0, self.window_size - len(chunk)))
            prob = model(chunk.unsqueeze(0), self.sr).item()
            probs.append(prob)

        # Enrich segments with per-segment probability stats
        rich_segments = []
        prob_array = np.array(probs)
        for s in segments:
            start_idx = int(s["start"] / self.window_size)
            end_idx = int(s["end"] / self.window_size)
            seg_probs = prob_array[start_idx:end_idx]

            seg = SpeechSegment(
                start_sec=round(s["start"] / self.sr, 3),
                end_sec=round(s["end"] / self.sr, 3),
                duration_sec=round((s["end"] - s["start"]) / self.sr, 3),
                avg_probability=round(float(seg_probs.mean()), 3) if len(seg_probs) > 0 else 0.0,
                min_probability=round(float(seg_probs.min()), 3) if len(seg_probs) > 0 else 0.0,
                max_probability=round(float(seg_probs.max()), 3) if len(seg_probs) > 0 else 0.0,
                std_probability=round(float(seg_probs.std()), 3) if len(seg_probs) > 0 else 0.0,
                percent_above_threshold=round(
                    sum(p > self.threshold for p in seg_probs) / len(seg_probs) * 100, 1
                ) if len(seg_probs) > 0 else 0.0,
            )
            rich_segments.append(seg)

        return probs, rich_segments

    def plot_insights(
        self,
        probs: List[float],
        segments: List[SpeechSegment],
        audio_path: str | Path,
        out_dir: str | Path,
    ) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        time_axis = [i * self.step_sec for i in range(len(probs))]
        total_sec = len(probs) * self.step_sec

        # Improved dynamic x-axis labeling with denser ticks and max ~12 labels
        def set_dynamic_xticks(ax, total_seconds: float) -> None:
            """Set x-ticks with adaptive interval to avoid crowding while reducing large gaps."""
            import numpy as np

            # Target ~10–14 ticks max for good readability across durations
            target_ticks = 12
            interval = max(1, round(total_seconds / target_ticks, -1))  # round to nearest 10s base

            # Fine-tune for better human-readable intervals
            if interval <= 5:
                interval = 5
            elif interval <= 12:
                interval = 10
            elif interval <= 35:
                interval = 30
            elif interval <= 70:
                interval = 60
            elif interval <= 150:
                interval = 120
            else:
                interval = 300  # 5 min for very long files

            ticks = np.arange(0, total_seconds + interval / 2, interval)  # slight overrun ok
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

        plt.figure(figsize=(20, 10))
        plt.suptitle(f"Silero VAD Full Analysis – {Path(audio_path).name}", fontsize=18, fontweight="bold")

        # Top: Probability curve
        plt.subplot(2, 1, 1)
        plt.fill_between(time_axis, probs, color="skyblue")
        plt.axhline(self.threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {self.threshold}")
        plt.axhline(self.threshold - 0.15, color="orange", linestyle=":", linewidth=1.5, label="Negative Threshold")
        plt.ylim(0, 1.05)
        plt.xlim(0, total_sec)
        plt.ylabel("Probability", fontsize=14)
        plt.title("Raw Speech Probability Over Time", fontsize=14)
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        set_dynamic_xticks(plt.gca(), total_sec)

        # Bottom: Detected speech segments
        ax2 = plt.subplot(2, 1, 2)
        for i, seg in enumerate(segments):
            ax2.fill_between(
                [seg.start_sec, seg.end_sec],
                0,
                1,
                color="lightgreen",
                edgecolor="black",
                linewidth=1,
                alpha=0.8,
                label="Speech" if i == 0 else None,
            )
            mid = (seg.start_sec + seg.end_sec) / 2
            # Dark text for better readability on lightgreen background
            ax2.text(
                mid, 0.5, f"{seg.duration_sec:.1f}s",
                ha="center", va="center",
                color="darkgreen",  # dark and high contrast
                fontweight="bold",
                fontsize=10,
            )
        set_dynamic_xticks(ax2, total_sec)

        plt.xlim(0, total_sec)
        plt.ylim(0, 1)
        plt.xlabel("Time (seconds)", fontsize=14)
        plt.title(f"Detected Speech Segments (n={len(segments)})", fontsize=14)
        if segments:
            plt.legend()

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.subplots_adjust(hspace=0.3)

        # Save
        plot_path = Path(out_dir) / f"vad_detected_speech_{Path(audio_path).stem}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved → {plot_path}")

        plt.close()

        # Apply dynamic ticks to histogram as well (optional but nice)
        plt.figure(figsize=(12, 8))
        prob_array = np.array(probs)
        speech_probs = []
        silence_probs = []
        for seg in segments:
            start_idx = int(seg.start_sec / self.step_sec)
            end_idx = int(seg.end_sec / self.step_sec)
            speech_probs.extend(prob_array[start_idx:end_idx])
        speech_probs = np.array(speech_probs)
        silence_mask = np.ones(len(prob_array), dtype=bool)
        for seg in segments:
            start_idx = int(seg.start_sec / self.step_sec)
            end_idx = int(seg.end_sec / self.step_sec)
            silence_mask[start_idx:end_idx] = False
        silence_probs = prob_array[silence_mask]
        plt.hist(silence_probs, bins=50, alpha=0.6, label="Silence windows", color="lightgray", density=True)
        plt.hist(speech_probs, bins=50, alpha=0.7, label="Speech windows", color="lightgreen", density=True)
        plt.axvline(self.threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {self.threshold}")
        plt.xlabel("Speech Probability")
        plt.ylabel("Density")
        plt.title("Probability Distribution (Speech vs. Silence)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        hist_path = Path(out_dir) / f"vad_probability_histogram_{Path(audio_path).stem}.png"
        plt.savefig(hist_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Histogram saved → {hist_path}")

        # Apply to segment confidence chart (x-axis is probability 0-1, no need for time ticks)
        if segments:
            plt.figure(figsize=(12, max(6, len(segments) * 0.4)))
            avg_probs = [seg.avg_probability for seg in segments]
            durations = [seg.duration_sec for seg in segments]
            indices = np.arange(len(segments))
            colors = ["green" if p > 0.8 else "orange" if p > 0.6 else "red" for p in avg_probs]
            bars = plt.barh(indices, avg_probs, color=colors, alpha=0.8)
            plt.yticks(indices, [f"Seg {i+1} ({d:.1f}s)" for i, d in enumerate(durations)])
            plt.xlabel("Average Probability")
            plt.title("Per-Segment Confidence (Avg Probability)")
            plt.xlim(0, 1)
            plt.xticks(np.linspace(0, 1, 11))  # fixed 0.0 to 1.0 with 0.1 steps
            # Dark text annotations on bars
            for bar, prob in zip(bars, avg_probs):
                plt.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{prob:.3f}",
                    va="center",
                    fontsize=9,
                    color="black",  # high contrast regardless of bar color
                    fontweight="bold",
                )
            plt.grid(True, alpha=0.3, axis="x")
            conf_path = Path(out_dir) / f"vad_segment_confidence_{Path(audio_path).stem}.png"
            plt.savefig(conf_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Segment confidence chart saved → {conf_path}")

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

    # NEW: Save each segment as individual .wav + meta.json
    def save_segments_individually(
        self,
        audio_path: str | Path,
        segments: List[SpeechSegment],
        out_dir: str | Path,
    ) -> None:
        """Extract and save each speech segment as a separate .wav file with accompanying meta.json."""
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        wav, sr = sf.read(str(audio_path))
        if sr != self.sr:
            raise ValueError(f"Audio sampling rate {sr} does not match analyzer's {self.sr}")
        for idx, seg in enumerate(segments, start=1):
            seg_dir = Path(out_dir) / f"segment_{idx:03d}"
            seg_dir.mkdir(parents=True, exist_ok=True)

            start_sample = int(seg.start_sec * self.sr)
            end_sample = int(seg.end_sec * self.sr)
            segment_audio = wav[start_sample:end_sample]

            wav_path = seg_dir / "sound.wav"
            sf.write(str(wav_path), segment_audio, self.sr)

            meta = {
                "segment_index": idx,
                "start_sec": seg.start_sec,
                "end_sec": seg.end_sec,
                "duration_sec": seg.duration_sec,
                "original_file": Path(audio_path).name,
                "avg_probability": seg.avg_probability,
                "min_probability": seg.min_probability,
                "max_probability": seg.max_probability,
                "std_probability": seg.std_probability,
                "percent_above_threshold": seg.percent_above_threshold,
            }
            meta_path = seg_dir / "meta.json"
            meta_path.write_text(json.dumps(meta, indent=2))

        print(f"Saved {len(segments)} individual segments → {out_dir}")

    def get_metrics(
        self,
        probs: List[float],
        segments: List[SpeechSegment],
        total_duration_sec: float,
    ) -> dict:
        """Return a clean dictionary of all useful metrics."""
        durations = [s.duration_sec for s in segments]
        num_segments = len(segments)
        total_speech_sec = sum(durations)
        speech_percent = total_speech_sec / total_duration_sec * 100 if total_duration_sec else 0

        # Confidence
        prob_array = np.array(probs)
        speech_mask = np.zeros(len(probs), dtype=bool)
        for seg in segments:
            start_idx = int(seg.start_sec / self.step_sec)
            end_idx = int(seg.end_sec / self.step_sec)
            speech_mask[start_idx:end_idx] = True
        avg_prob_speech = float(prob_array[speech_mask].mean()) if speech_mask.any() else 0.0
        avg_prob_silence = float(prob_array[~speech_mask].mean()) if (~speech_mask).any() else 0.0

        # Gaps between segments
        gaps = (
            [segments[i + 1].start_sec - segments[i].end_sec for i in range(num_segments - 1)]
            if num_segments > 1
            else []
        )
        avg_gap_sec = float(np.mean(gaps)) if gaps else 0.0

        return {
            "total_duration_sec": round(total_duration_sec, 3),
            "total_speech_sec": round(total_speech_sec, 3),
            "speech_percentage": round(speech_percent, 1),
            "total_silence_sec": round(total_duration_sec - total_speech_sec, 3),
            "num_segments": num_segments,
            "avg_segment_sec": round(np.mean(durations), 3) if durations else 0,
            "median_segment_sec": round(np.median(durations), 3) if durations else 0,
            "shortest_segment_sec": round(min(durations), 3) if durations else 0,
            "longest_segment_sec": round(max(durations), 3) if durations else 0,
            "speech_rate_per_minute": round(num_segments / (total_duration_sec / 60), 1)
            if total_duration_sec
            else 0,
            "avg_gap_between_segments_sec": round(avg_gap_sec, 3),
            "avg_probability_all": round(float(prob_array.mean()), 3),
            "avg_probability_in_speech": round(avg_prob_speech, 3),
            "avg_probability_in_silence": round(avg_prob_silence, 3),
            "windows_above_threshold_percent": round(
                sum(p > self.threshold for p in probs) / len(probs) * 100, 1
            ),
            "median_probability_all": round(float(np.median(prob_array)), 3),
            "p95_probability_in_speech": round(float(np.percentile(prob_array[speech_mask], 95)), 3)
            if speech_mask.any() else 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description="Silero VAD Full Analyzer – Beautiful insights + JSON")
    parser.add_argument("audio", type=Path, help="Path to input .wav file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("vad_results"),
        help="Output directory (default: ./vad_results)",
    )
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="VAD threshold (0.3–0.7 typical)")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Audio not found: {args.audio}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = SileroVADAnalyzer(
        threshold=args.threshold,
        min_speech_duration_ms=250,
        min_silence_duration_ms=120,
        speech_pad_ms=30,
    )

    print(f"Analyzing: {args.audio.name}")
    print(f"Threshold: {args.threshold} | Output → {args.output_dir.resolve()}")

    probs, segments = analyzer.analyze(args.audio)
    
    total_sec = len(probs) * analyzer.step_sec
    metrics = analyzer.get_metrics(probs, segments, total_sec)
    
    analyzer.plot_insights(probs, segments, args.audio, args.output_dir)
    analyzer.save_json(segments, args.output_dir, args.audio)
    analyzer.save_segments_individually(args.audio, segments, Path(args.output_dir) / "segments")

    # NEW: Pretty table in console
    from rich.table import Table
    from rich.console import Console
    console = Console()
    table = Table(title=f"[bold]VAD Metrics – {args.audio.name}[/bold]")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in metrics.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)

    # NEW: Save metrics as JSON too
    metrics_path = args.output_dir / f"vad_metrics_{Path(args.audio).stem}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics JSON → {metrics_path}")

    print("\nAll done! Open the PNGs to see everything at a glance.")
    print(f"→ {args.output_dir}")


if __name__ == "__main__":
    main()