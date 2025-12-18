# jet_python_modules/jet/audio/speech/silero/silero_vad_file.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.logging import RichHandler
import logging

from jet.audio.speech.silero.speech_timestamps_extractor import extract_speech_timestamps
from jet.audio.speech.silero.speech_types import SpeechSegment as ExtractedSegment
from silero_vad.utils_vad import save_audio, read_audio

log = logging.getLogger(__name__)
log.handlers = []  # Avoid duplicate handlers
if not log.handlers:
    log.addHandler(RichHandler(rich_tracebacks=True, markup=True))
    log.setLevel(logging.INFO)


@dataclass
class SpeechSegment:
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    duration_sec: float

    def duration(self) -> float:
        return self.duration_sec


class SileroVADFileProcessor:
    def __init__(
        self,
        threshold: float = 0.6,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 700,
        speech_pad_ms: int = 30,
        min_speech_duration_ms: int = 500,
        output_dir: Optional[Path | str] = None,
        save_segments: bool = True,
        debug: bool = False,
        on_speech_start: Optional[Callable[[float], None]] = None,
        on_speech_end: Optional[Callable[[SpeechSegment], None]] = None,
    ):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_segments = save_segments and bool(self.output_dir)
        self.debug = debug
        if debug:
            log.setLevel(logging.DEBUG)

        self.on_speech_start = on_speech_start or self._default_start_handler
        self.on_speech_end = on_speech_end or self._default_end_handler

        if self.save_segments:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._segment_counter = 0

    def _default_start_handler(self, timestamp: float) -> None:
        log.info(f"[green]Speech Start[/] @ {timestamp:.3f}s")

    def _default_end_handler(self, segment: SpeechSegment) -> None:
        log.info(
            f"[bold magenta]Speech End[/] @ {segment.end_sec:.3f}s "
            f"[cyan]dur={segment.duration():.3f}s[/]"
        )

    def process(self, audio_path: str | Path) -> List[SpeechSegment]:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        log.info(f"[bold blue]Loading audio:[/] {audio_path.name}")
        audio = read_audio(str(audio_path), sampling_rate=self.sampling_rate)

        log.info("[bold blue]Running VAD on file...[/]")
        raw_segments: List[ExtractedSegment] = extract_speech_timestamps(
            audio=audio,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,
        )

        if not raw_segments:
            log.warning("[yellow]No speech segments detected[/]")
            return []

        log.info(f"[bold green]Found {len(raw_segments)} speech segment(s)[/]")

        results: List[SpeechSegment] = []
        for idx, seg in enumerate(raw_segments):
            start_sample = seg["start"]
            end_sample = seg["end"]
            start_sec = start_sample / self.sampling_rate
            end_sec = end_sample / self.sampling_rate
            duration_sec = end_sec - start_sec

            segment = SpeechSegment(
                start_sample=start_sample,
                end_sample=end_sample,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=duration_sec,
            )

            self.on_speech_start(start_sec)

            audio_tensor = audio[start_sample:end_sample]
            probabilities = self._collect_probabilities_in_segment(audio, start_sample, end_sample)
            energies = self._compute_energies(audio_tensor)
            strong_chunks, weak_chunks = self._find_confidence_chunks(probabilities)

            avg_prob = np.mean([p for _, p in probabilities]) if probabilities else 0.0
            rms_energy = np.sqrt(np.mean(audio_tensor.numpy() ** 2))

            log.info(
                f"[bold magenta]Segment {idx + 1}[/] | "
                f"{start_sec:.2f} → {end_sec:.2f}s | "
                f"dur={duration_sec:.2f}s | "
                f"prob={avg_prob:.3f} | rms={rms_energy:.5f}"
            )

            if self.save_segments:
                self._segment_counter += 1
                seg_dir = self.output_dir / f"segment_{self._segment_counter:03d}"
                seg_dir.mkdir(parents=True, exist_ok=True)

                # Save audio
                wav_path = seg_dir / "sound.wav"
                save_audio(str(wav_path), audio_tensor.unsqueeze(0), self.sampling_rate)

                # Save metadata
                (seg_dir / "segment.json").write_text(
                    json.dumps({
                        "idx": idx,
                        "start_sec": round(start_sec, 4),
                        "end_sec": round(end_sec, 4),
                        "duration_sec": round(duration_sec, 4),
                        "avg_probability": round(avg_prob, 4),
                        "rms_energy": round(float(rms_energy), 6),
                        "source_file": str(audio_path),
                    }, indent=2)
                )

                # Save probabilities & energy
                (seg_dir / "probabilities.json").write_text(
                    json.dumps([{"time_sec": round(t - start_sec, 3), "probability": p} for t, p in probabilities], indent=2)
                )
                (seg_dir / "energy.json").write_text(
                    json.dumps([{"time_sec": round(t - start_sec, 3), "rms": e} for t, e in energies], indent=2)
                )

                # Save strong/weak chunks
                strong_list = [(c["start_sec"], c["end_sec"]) for c in strong_chunks]
                weak_list = [(c["start_sec"], c["end_sec"]) for c in weak_chunks]
                (seg_dir / "strong_chunks.json").write_text(json.dumps(strong_chunks, indent=2))
                (seg_dir / "weak_chunks.json").write_text(json.dumps(weak_chunks, indent=2))

                # Visualizations
                self._save_segment_visualization(
                    audio_tensor=audio_tensor,
                    probabilities=[(t - start_sec, p) for t, p in probabilities],
                    energies=energies,
                    seg_dir=seg_dir,
                    strong_chunks=strong_list,
                    weak_chunks=weak_list,
                )

                log.info(f"[bold green]Saved segment:[/] {seg_dir.name}")

            self.on_speech_end(segment)
            results.append(segment)

        return results

    def _collect_probabilities_in_segment(
        self,
        full_audio: torch.Tensor,
        start_sample: int,
        end_sample: int,
    ) -> List[Tuple[float, float]]:
        # Re-run model only on segment window for accurate per-frame probabilities
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )
        window = 512 if self.sampling_rate == 16000 else 256
        probs = []
        segment_audio = full_audio[start_sample:end_sample]
        for i in range(0, len(segment_audio), window):
            chunk = segment_audio[i:i + window]
            if len(chunk) < window:
                chunk = torch.nn.functional.pad(chunk, (0, window - len(chunk)))
            prob = model(chunk.unsqueeze(0), self.sampling_rate).item()
            time_sec = (start_sample + i + window // 2) / self.sampling_rate
            probs.append((time_sec, prob))
        return probs

    def _compute_energies(self, audio_tensor: torch.Tensor) -> List[Tuple[float, float]]:
        window_samples = int(0.02 * self.sampling_rate)
        hop = window_samples // 2
        energies = []
        audio_np = audio_tensor.numpy()
        base_time = audio_tensor.abs().sum().item() > 0  # dummy to get start time later
        for start in range(0, len(audio_np), hop):
            end = start + window_samples
            window = audio_np[start:end]
            if len(window) < window_samples:
                window = np.pad(window, (0, window_samples - len(window)))
            rms = np.sqrt(np.mean(window ** 2))
            timestamp_sec = start / self.sampling_rate
            energies.append((round(timestamp_sec, 3), float(rms)))
        return energies

    def _find_confidence_chunks(
        self,
        prob_in_segment: List[Tuple[float, float]],
        strong_threshold: float = 0.85,
        weak_threshold: float = 0.60,
        min_duration_sec: float = 0.2,
    ) -> Tuple[List[dict], List[dict]]:
        # Same logic as streamer
        if not prob_in_segment:
            return [], []

        times, probs = zip(*prob_in_segment)
        times = np.array(times)
        probs = np.array(probs)
        min_frames = max(1, int(min_duration_sec * self.sampling_rate / 512))

        strong_chunks = []
        weak_chunks = []
        current = None

        for t, p in prob_in_segment:
            if p >= strong_threshold:
                if current and current["type"] == "strong":
                    current["end_t"] = t
                    current["frames"].append((t, p))
                else:
                    if current and len(current["frames"]) >= min_frames:
                        (strong_chunks if current["type"] == "strong" else weak_chunks).append(current)
                    current = {"type": "strong", "start_t": t, "end_t": t, "frames": [(t, p)]}
            elif p <= weak_threshold:
                if current and current["type"] == "weak":
                    current["end_t"] = t
                    current["frames"].append((t, p))
                else:
                    if current and len(current["frames"]) >= min_frames:
                        (strong_chunks if current["type"] == "strong" else weak_chunks).append(current)
                    current = {"type": "weak", "start_t": t, "end_t": t, "frames": [(t, p)]}
            else:
                if current and len(current["frames"]) >= min_frames:
                    (strong_chunks if current["type"] == "strong" else weak_chunks).append(current)
                current = None

        if current and len(current["frames"]) >= min_frames:
            (strong_chunks if current["type"] == "strong" else weak_chunks).append(current)

        def finalize(chunks):
            return [{
                "start_sec": c["start_t"],
                "end_sec": c["end_t"],
                "duration_sec": round(c["end_t"] - c["start_t"], 3),
                "avg_probability": round(np.mean([p for _, p in c["frames"]]), 4),
                "peak_probability": round(max(p for _, p in c["frames"]), 4),
            } for c in chunks]

        return finalize(strong_chunks), finalize(weak_chunks)

    def _configure_time_axis(self, ax, duration: float) -> None:
        from matplotlib.ticker import MaxNLocator
        ax.set_xlim(0, duration)
        if duration <= 0.0:
            return
        target_ticks = max(12, min(20, int(duration / 0.5) + 1))
        locator = MaxNLocator(nbins=target_ticks, steps=[1, 2, 4, 5, 10], min_n_ticks=8)
        ax.xaxis.set_major_locator(locator)
        forced = {0.0, round(duration, 6)}
        ax.set_xticks(sorted(set(ax.get_xticks()).union(forced)))
        ax.set_xlabel("Time (seconds)", fontsize=12)

    def _save_segment_visualization(
        self,
        audio_tensor: torch.Tensor,
        probabilities: List[Tuple[float, float]],
        energies: List[Tuple[float, float]],
        seg_dir: Path,
        strong_chunks: List[Tuple[float, float]],
        weak_chunks: List[Tuple[float, float]],
    ) -> None:
        duration = len(audio_tensor) / self.sampling_rate
        title_suffix = f"Duration: {duration:.2f}s"

        plt.style.use("seaborn-v0_8-whitegrid")

        # Waveform
        fig, ax = plt.subplots(figsize=(16, 5))
        times = np.linspace(0, duration, len(audio_tensor))
        ax.plot(times, audio_tensor.numpy(), color="#1f77b4")
        self._configure_time_axis(ax, duration)
        ax.set_title(f"Waveform – {title_suffix}", fontsize=16)
        ax.set_ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(seg_dir / "waveform.png", dpi=300)
        plt.close(fig)

        # VAD Probability
        if probabilities:
            fig, ax = plt.subplots(figsize=(16, 5))
            ts, ps = zip(*probabilities)
            ax.plot(ts, ps, color="#ff7f0e", linewidth=2)
            ax.axhline(self.threshold, color="red", linestyle="--", label=f"Threshold = {self.threshold}")
            self._configure_time_axis(ax, duration)
            ax.set_ylim(0, 1.1)
            ax.set_title(f"VAD Probability – {title_suffix}", fontsize=16)
            ax.legend()
            plt.tight_layout()
            plt.savefig(seg_dir / "vad_probability.png", dpi=300)
            plt.close(fig)

        # Energy
        if energies:
            fig, ax = plt.subplots(figsize=(16, 5))
            ts, es = zip(*energies)
            ax.plot(ts, es, color="#2ca02c")
            self._configure_time_axis(ax, duration)
            ax.set_title(f"Energy Envelope – {title_suffix}", fontsize=16)
            ax.set_ylabel("RMS Energy")
            plt.tight_layout()
            plt.savefig(seg_dir / "energy.png", dpi=300)
            plt.close(fig)


if __name__ == "__main__":
    import os
    import shutil

    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    processor = SileroVADFileProcessor(
        output_dir=OUTPUT_DIR,
        save_segments=True,
        debug=True,
    )

    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"
    segments = processor.process(audio_file)
    log.info(f"[bold green]Processing complete. {len(segments)} segments saved to {OUTPUT_DIR}[/]")