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

import matplotlib.pyplot as plt
import torch

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
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    duration_sec: float


class SileroVADAnalyzer:
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 500,
        min_silence_duration_ms: int = 700,
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

        # Convert segments to rich objects
        rich_segments = [
            SpeechSegment(
                start_sample=s["start"],
                end_sample=s["end"],
                start_sec=round(s["start"] / self.sr, 3),
                end_sec=round(s["end"] / self.sr, 3),
                duration_sec=round((s["end"] - s["start"]) / self.sr, 3),
            )
            for s in segments
        ]

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

        plt.figure(figsize=(20, 10))
        plt.suptitle(f"Silero VAD Full Analysis – {Path(audio_path).name}", fontsize=18, fontweight="bold")

        # Top: Probability curve
        plt.subplot(2, 1, 1)
        plt.fill_between(time_axis, probs, color="#4CAF50", alpha=0.7, label="Speech Probability")
        plt.axhline(self.threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {self.threshold}")
        plt.axhline(self.threshold - 0.15, color="orange", linestyle=":", linewidth=1.5, label="Negative Threshold")
        plt.ylim(0, 1.05)
        plt.xlim(0, total_sec)
        plt.ylabel("Probability", fontsize=14)
        plt.title("Raw Speech Probability Over Time", fontsize=14)
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

        # Bottom: Detected speech segments
        ax2 = plt.subplot(2, 1, 2)
        for i, seg in enumerate(segments):
            ax2.fill_between(
                [seg.start_sec, seg.end_sec],
                0,
                1,
                color="#2196F3",
                edgecolor="black",
                linewidth=1,
                alpha=0.8,
                label="Speech" if i == 0 else None,
            )
            mid = (seg.start_sec + seg.end_sec) / 2
            ax2.text(mid, 0.5, f"{seg.duration_sec}s", ha="center", va="center", color="white", fontweight="bold")

        plt.xlim(0, total_sec)
        plt.ylim(0, 1)
        plt.xlabel("Time (seconds)", fontsize=14)
        plt.title(f"Detected Speech Segments (n={len(segments)})", fontsize=14)
        if segments:
            plt.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(hspace=0.3)

        # Save
        plot_path = Path(out_dir) / f"vad_analysis_{Path(audio_path).stem}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved → {plot_path}")

        # Also save probability as separate high-res
        plt.figure(figsize=(18, 6))
        plt.plot(time_axis, probs, color="#2E7D32", linewidth=1.5)
        plt.fill_between(time_axis, probs, color="#81C784", alpha=0.6)
        plt.axhline(self.threshold, color="red", linestyle="--", linewidth=2)
        plt.title("Speech Probability Only (zoomable)")
        plt.xlabel("Seconds")
        plt.ylabel("Probability")
        prob_path = Path(out_dir) / f"vad_probability_{Path(audio_path).stem}.png"
        plt.savefig(prob_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        print(f"Probability plot → {prob_path}")

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
    analyzer.plot_insights(probs, segments, args.audio, args.output_dir)
    analyzer.save_json(segments, args.output_dir, args.audio)

    print("\nAll done! Open the PNGs to see everything at a glance.")
    print(f"→ {args.output_dir}")


if __name__ == "__main__":
    main()