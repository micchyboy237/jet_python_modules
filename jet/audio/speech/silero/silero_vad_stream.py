# silero_vad_stream.py
from __future__ import annotations
import signal
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, List, Tuple

import sounddevice as sd
import torch
import json
import logging
from rich.logging import RichHandler

# === NEW IMPORTS ===
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ────────────── Add Transcription Pipeline Import ──────────────
from jet.audio.transcribers.transcription_pipeline import TranscriptionPipeline

# ────────────── Logging Setup (replacing global Console) ──────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("silero_vad")

model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True,
    verbose=False,
)

(
    get_speech_timestamps,
    save_audio,
    _,
    VADIterator,
    _,
) = utils

@dataclass
class SpeechSegment:
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    duration_sec: float

    def duration(self) -> float:
        return self.duration_sec

# # ────────────── Instantiate Transcription Pipeline (Global) ──────────────
# trans_pipeline = TranscriptionPipeline(max_workers=3)

class SileroVADStreamer:
    def __init__(
        self,
        threshold: float = 0.6,
        sample_rate: int = 16000,
        min_silence_duration_ms: int = 700,
        speech_pad_ms: int = 30,
        device: Optional[int] = None,
        block_size: int = 512,
        on_speech_start: Optional[Callable[[float], None]] = None,
        on_speech_end: Optional[Callable[[SpeechSegment], None]] = None,
        output_dir: Optional[Path | str] = None,
        save_segments: bool = True,
        debug: bool = False,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.device = device
        self.on_speech_start = on_speech_start or self._default_start_handler
        self.on_speech_end = on_speech_end or self._default_end_handler
        self.debug = debug
        if debug:
            log.setLevel(logging.DEBUG)

        # Saving options
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_segments = save_segments and bool(self.output_dir)
        if self.save_segments:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._segment_counter = 0

        self.vad_iterator = VADIterator(
            model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        # Streaming state
        self._current_start: Optional[float] = None
        self._current_start_sample: Optional[int] = None
        self._total_samples: int = 0
        self._audio_buffer = deque()  # holds torch tensors (float32, mono)
        self._prob_buffer = deque()
        self._prob_timestamps = deque()

        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    def _default_start_handler(self, timestamp: float) -> None:
        log.info(f"[green]Speech Start[/] @ {timestamp:.3f}s")

    def _default_end_handler(self, segment: SpeechSegment) -> None:
        log.info(
            f"[bold magenta]Speech End[/] @ {segment.end_sec:.3f}s "
            f"[cyan]dur={segment.duration():.3f}s[/]"
        )

    def _configure_time_axis(self, ax, duration: float) -> None:
        """
        Smart X-axis tick placement:
        • Minimum tick spacing ≈ 0.5 s
        • Dynamically widens for longer segments (0.5 → 1 → 2 → 5 s steps)
        • Always includes 0.0 and the exact end time
        • No duplicate labels
        """
        from matplotlib.ticker import MaxNLocator
        import matplotlib.pyplot as plt

        ax.set_xlim(0, duration)

        if duration <= 0.0:
            return

        # Very short clips → fine, fixed grid
        if duration < 1.0:
            step = 0.1 if duration >= 0.4 else 0.05
            ax.xaxis.set_major_locator(plt.MultipleLocator(step))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(step / 5))
        else:
            # Dynamically determine minimum tick gap for longer audio.
            # Target: 12–20 major ticks; enforces >=0.5s min between.
            target_ticks = max(12, min(20, int(duration / 0.5) + 1))
            locator = MaxNLocator(
                nbins=target_ticks,
                steps=[1, 2, 4, 5, 10],
                min_n_ticks=8,
                integer=False,
            )
            ax.xaxis.set_major_locator(locator)

        # Always force 0.0 and exact end (rounded) to appear (avoids missing rightmost label)
        current_ticks = set(ax.get_xticks())
        forced = {0.0, round(duration, 6)}
        ax.set_xticks(sorted(current_ticks.union(forced)))

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
        duration = len(audio_tensor) / self.sample_rate
        title_suffix = f"Segment Duration: {duration:.2f}s"

        self._save_waveform_plot(audio_tensor, seg_dir, title_suffix)
        self._save_energy_plot(energies, seg_dir, title_suffix)

        # New split VAD charts
        self._save_vad_probability_clean(probabilities, seg_dir, title_suffix)
        self._save_vad_strong_regions(probabilities, strong_chunks, seg_dir, title_suffix)
        self._save_vad_weak_regions(probabilities, weak_chunks, seg_dir, title_suffix)

    def _save_waveform_plot(self, audio_tensor: torch.Tensor, seg_dir: Path, title_suffix: str) -> None:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(16, 6))
        duration = len(audio_tensor) / self.sample_rate
        times = np.linspace(0, duration, len(audio_tensor), endpoint=True)
        ax.plot(times, audio_tensor.numpy(), color="#1f77b4", linewidth=0.8)
        self._configure_time_axis(ax, duration)
        ax.set_title(f"Waveform – {title_suffix}", fontsize=16, pad=20)
        ax.set_ylabel("Amplitude", fontsize=12)
        ax.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(seg_dir / "waveform.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _save_vad_probability_clean(
        self,
        probabilities: List[Tuple[float, float]],
        seg_dir: Path,
        title_suffix: str,
    ) -> None:
        if not probabilities:
            return
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(16, 6))
        ts, probs = zip(*probabilities)
        duration = ts[-1] if ts else 0.0

        ax.plot(ts, probs, color="#6a0dad", linewidth=2.5, label="Speech Probability")
        ax.fill_between(ts, 0, probs, color="#6a0dad", alpha=0.25)
        ax.axhline(y=self.threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {self.threshold:.2f}")

        self._configure_time_axis(ax, duration)

        ax.set_ylim(0, 1.1)
        ax.set_title(f"VAD Probability (Clean) – {title_suffix}", fontsize=16, pad=20)
        ax.set_ylabel("Probability", fontsize=12)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(seg_dir / "vad_probability_clean.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _save_vad_strong_regions(
        self,
        probabilities: List[Tuple[float, float]],
        strong_chunks: List[Tuple[float, float]],
        seg_dir: Path,
        title_suffix: str,
    ) -> None:
        if not probabilities:
            return
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(16, 6))
        ts, probs = zip(*probabilities)
        duration = ts[-1] if ts else 0.0

        ax.plot(ts, probs, color="#2ca02c", linewidth=1.2, alpha=0.6)
        for i, (start, end) in enumerate(strong_chunks):
            ax.axvspan(start, end, color="#2ca02c", alpha=0.6,
                       label="Strong Confidence" if i == 0 else "")

        self._configure_time_axis(ax, duration)

        ax.set_ylim(0, 1.1)
        ax.set_title(f"Strong Confidence Regions (≥0.85) – {title_suffix}", fontsize=16, pad=20)
        ax.set_ylabel("Probability", fontsize=12)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(seg_dir / "vad_strong_confidence.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _save_vad_weak_regions(
        self,
        probabilities: List[Tuple[float, float]],
        weak_chunks: List[Tuple[float, float]],
        seg_dir: Path,
        title_suffix: str,
    ) -> None:
        if not probabilities:
            return
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(16, 6))
        ts, probs = zip(*probabilities)
        duration = ts[-1] if ts else 0.0

        ax.plot(ts, probs, color="#ff7f0e", linewidth=1.2, alpha=0.6)
        for i, (start, end) in enumerate(weak_chunks):
            ax.axvspan(start, end, color="#ff7f0e", alpha=0.55,
                       label="Weak / Uncertain" if i == 0 else "")

        self._configure_time_axis(ax, duration)

        ax.set_ylim(0, 1.1)
        ax.set_title(f"Weak / Uncertain Regions (≤0.60) – {title_suffix}", fontsize=16, pad=20)
        ax.set_ylabel("Probability", fontsize=12)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(seg_dir / "vad_weak_confidence.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _save_energy_plot(self, energies: List[Tuple[float, float]], seg_dir: Path, title_suffix: str) -> None:
        if not energies:
            return
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(16, 6))
        ts, rms_vals = zip(*energies)
        duration = ts[-1] if ts else 0.0

        ax.plot(ts, rms_vals, color="#ff7f0e", linewidth=2, label="20ms RMS Energy")

        self._configure_time_axis(ax, duration)

        ax.set_title(f"Energy Envelope – {title_suffix}", fontsize=16, pad=20)
        ax.set_ylabel("RMS Energy", fontsize=12)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(seg_dir / "energy.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # === The remainder of the methods below are unchanged ===

    def _compute_energies(self, audio_tensor: torch.Tensor) -> List[Tuple[float, float]]:
        window_samples = int(0.02 * self.sample_rate)
        energies = []
        hop = window_samples // 2
        audio_np = audio_tensor.numpy()
        for start in range(0, len(audio_np), hop):
            end = start + window_samples
            window = audio_np[start:end]
            if len(window) < window_samples:
                window = np.pad(window, (0, window_samples - len(window)))
            rms = np.sqrt(np.mean(window ** 2))
            timestamp_sec = start / self.sample_rate + self._current_start
            energies.append((round(timestamp_sec, 3), float(rms)))
        return energies

    def _save_chunk_clip(
        self,
        audio_tensor: torch.Tensor,
        start_sample: int,
        end_sample: int,
        label: str,
        chunk_idx: int,
        seg_dir: Path,
    ) -> Path:
        chunk_tensor = audio_tensor[start_sample:end_sample]
        chunk_dir = seg_dir / f"{label}_chunks"
        chunk_dir.mkdir(exist_ok=True)
        path = chunk_dir / f"{label}_{chunk_idx:03d}.wav"
        save_audio(str(path), chunk_tensor.unsqueeze(0), self.sample_rate)
        return path

    def _find_confidence_chunks(
        self,
        prob_in_segment: List[Tuple[float, float]],
        strong_threshold: float = 0.85,
        weak_threshold: float = 0.60,
        min_duration_sec: float = 0.2,
    ) -> Tuple[List[dict], List[dict]]:
        if not prob_in_segment:
            return [], []

        times, probs = zip(*prob_in_segment)
        times = np.array(times)
        probs = np.array(probs)

        min_frames = int(min_duration_sec * self.sample_rate / self.block_size)

        strong_chunks = []
        weak_chunks = []
        current_chunk = None

        for i, (t, p) in enumerate(prob_in_segment):
            frame_center_sample = int((t) * self.sample_rate)

            if p >= strong_threshold:
                if current_chunk and current_chunk["type"] == "strong":
                    current_chunk["end_t"] = t
                    current_chunk["end_sample"] = frame_center_sample + self.block_size // 2
                    current_chunk["frames"].append((t, p))
                else:
                    if current_chunk and len(current_chunk["frames"]) >= min_frames:
                        if current_chunk["type"] == "strong":
                            strong_chunks.append(current_chunk)
                        else:
                            weak_chunks.append(current_chunk)
                    current_chunk = {
                        "type": "strong",
                        "start_t": t,
                        "end_t": t,
                        "start_sample": frame_center_sample - self.block_size // 2,
                        "end_sample": frame_center_sample + self.block_size // 2,
                        "frames": [(t, p)],
                    }
            elif p <= weak_threshold:
                if current_chunk and current_chunk["type"] == "weak":
                    current_chunk["end_t"] = t
                    current_chunk["end_sample"] = frame_center_sample + self.block_size // 2
                    current_chunk["frames"].append((t, p))
                else:
                    if current_chunk and len(current_chunk["frames"]) >= min_frames:
                        if current_chunk["type"] == "strong":
                            strong_chunks.append(current_chunk)
                        else:
                            weak_chunks.append(current_chunk)
                    current_chunk = {
                        "type": "weak",
                        "start_t": t,
                        "end_t": t,
                        "start_sample": frame_center_sample - self.block_size // 2,
                        "end_sample": frame_center_sample + self.block_size // 2,
                        "frames": [(t, p)],
                    }
            else:
                if current_chunk and len(current_chunk["frames"]) >= min_frames:
                    if current_chunk["type"] == "strong":
                        strong_chunks.append(current_chunk)
                    else:
                        weak_chunks.append(current_chunk)
                current_chunk = None

        if current_chunk and len(current_chunk["frames"]) >= min_frames:
            if current_chunk["type"] == "strong":
                strong_chunks.append(current_chunk)
            else:
                weak_chunks.append(current_chunk)

        def finalize(chunks):
            return [{
                "start_sec": c["start_t"],
                "end_sec": c["end_t"],
                "duration_sec": round(c["end_t"] - c["start_t"], 3),
                "avg_probability": round(np.mean([p for _, p in c["frames"]]), 4),
                "peak_probability": round(max(p for _, p in c["frames"]), 4),
            } for c in chunks]

        return finalize(strong_chunks), finalize(weak_chunks)

    def _audio_callback(self, indata, frames, time, status):
        if status:
            log.warning(f"Audio warning: {status}")

        chunk = torch.from_numpy(indata.copy()).squeeze(1).float()
        current_time_sec = self._total_samples / self.sample_rate

        speech_prob = self.vad_iterator.model(chunk, self.sample_rate).item()

        with self._lock:
            self._audio_buffer.append(chunk)
            self._prob_buffer.append(speech_prob)
            self._prob_timestamps.append(current_time_sec + (len(chunk) / self.sample_rate) / 2)
            self._total_samples += len(chunk)

            result = self.vad_iterator(chunk, return_seconds=True)

        if self.debug:
            crossed_up   = speech_prob >= self.threshold and not getattr(self.vad_iterator, "triggered", False)
            crossed_down = speech_prob <  self.threshold and getattr(self.vad_iterator, "triggered", False)
            should_log_periodic = (
                not hasattr(self, "_last_debug_log")
                or (current_time_sec - self._last_debug_log) > 2.0
            )
            if crossed_up or crossed_down or should_log_periodic:
                log.debug(
                    f"[dim]Block @ {current_time_sec:6.2f}s | "
                    f"prob={speech_prob:.3f} {'[bold green]SPEECH[/]' if speech_prob >= self.threshold else '[bold red]SILENCE[/]'} | "
                    f"triggered={self.vad_iterator.triggered}[/]"
                )
                self._last_debug_log = current_time_sec

        if result is None:
            return

        if "start" in result:
            self._current_start = result["start"]
            self._current_start_sample = int(result["start"] * self.sample_rate)
            self.on_speech_start(result["start"])
            log.info(f"[green]Speech START[/] @ {result['start']:.3f}s | initial prob={speech_prob:.3f}")

        elif "end" in result and self._current_start is not None:
            end_sec = result["end"]
            end_sample_global = int(end_sec * self.sample_rate)

            start_sample = self._current_start_sample
            end_sample = end_sample_global

            duration_sec = end_sec - self._current_start

            segment = SpeechSegment(
                start_sample=start_sample,
                end_sample=end_sample,
                start_sec=self._current_start,
                end_sec=end_sec,
                duration_sec=duration_sec,
            )

            audio_tensor = self._extract_segment_audio(start_sample, end_sample)
            if audio_tensor is None:
                log.warning("Failed to extract audio for segment")
                return

            prob_in_segment = [
                (t - self._current_start, p)
                for t, p in zip(self._prob_timestamps, self._prob_buffer)
                if self._current_start <= t <= end_sec
            ]

            energies = self._compute_energies(audio_tensor)

            avg_prob = np.mean([p for _, p in prob_in_segment]) if prob_in_segment else 0.0
            max_prob = max((p for _, p in prob_in_segment), default=0.0)
            rms_energy = np.sqrt(np.mean(audio_tensor.numpy() ** 2))
            log.info(
                f"[bold magenta]Speech END[/] @ {end_sec:.3f}s | "
                f"dur={duration_sec:.3f}s | "
                f"avg_prob={avg_prob:.3f} | max_prob={max_prob:.3f} | "
                f"rms={rms_energy:.5f}"
            )

            # === CHUNK-LEVEL CONFIDENCE ANALYSIS ===
            strong_chunks, weak_chunks = self._find_confidence_chunks(prob_in_segment)

            if self.save_segments:
                self._segment_counter += 1
                seg_dir = Path(self.output_dir) / f"segment_{self._segment_counter:03d}"
                seg_dir.mkdir(parents=True, exist_ok=True)

                # Save audio & metadata
                wav_path = seg_dir / "sound.wav"
                json_path = seg_dir / "segment.json"
                save_audio(str(wav_path), audio_tensor, self.sample_rate)

                prob_path = seg_dir / "probabilities.json"
                energy_path = seg_dir / "energy.json"
                with prob_path.open("w") as f:
                    json.dump([{"time_sec": t, "probability": p} for t, p in prob_in_segment], f, indent=2)
                with energy_path.open("w") as f:
                    json.dump([{"time_sec": t, "rms_energy": e} for t, e in energies], f, indent=2)

                # Save strong/weak chunk audio clips + JSON
                for i, chunk in enumerate(strong_chunks):
                    start_s = int(chunk["start_sec"] * self.sample_rate)
                    end_s = int(chunk["end_sec"] * self.sample_rate)
                    self._save_chunk_clip(audio_tensor, start_s, end_s, "strong", i + 1, seg_dir)
                for i, chunk in enumerate(weak_chunks):
                    start_s = int(chunk["start_sec"] * self.sample_rate)
                    end_s = int(chunk["end_sec"] * self.sample_rate)
                    self._save_chunk_clip(audio_tensor, start_s, end_s, "weak", i + 1, seg_dir)
                (seg_dir / "strong_chunks.json").write_text(json.dumps(strong_chunks, indent=2))
                (seg_dir / "weak_chunks.json").write_text(json.dumps(weak_chunks, indent=2))

                # Save new, separate plots for waveform, probability, and energy (no combined plot)
                self._save_segment_visualization(
                    audio_tensor,
                    prob_in_segment,
                    energies,
                    seg_dir,
                    strong_chunks=[(c["start_sec"], c["end_sec"]) for c in strong_chunks],
                    weak_chunks=[(c["start_sec"], c["end_sec"]) for c in weak_chunks],
                )

                segment_data = {
                    "start_sample": segment.start_sample,
                    "end_sample": segment.end_sample,
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "duration_sec": segment.duration_sec,
                    "avg_speech_probability": round(avg_prob, 4),
                    "max_speech_probability": round(max_prob, 4),
                    "rms_energy": round(float(rms_energy), 6),
                    "num_probability_points": len(prob_in_segment),
                    "strong_chunk_count": len(strong_chunks),
                    "weak_chunk_count": len(weak_chunks),
                    "best_chunk_avg_prob": max((c["avg_probability"] for c in strong_chunks), default=0.0),
                    "worst_chunk_avg_prob": min((c["avg_probability"] for c in weak_chunks), default=1.0) if weak_chunks else None,
                }
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(segment_data, f, indent=2)

                log.info(f"[bold green]Saved rich segment:[/] {seg_dir.name} → audio | json | probabilities | energy | waveform.png/vad_probability.png/energy.png | strong/weak_chunks")

            # ────── ADD TRANSLATION PIPELINE SUBMIT HERE ──────
            # Convert torch → numpy (float32, mono) exactly as your transcriber expects
            audio_np: np.ndarray = audio_tensor.numpy().astype(np.float32)
            trans_pipeline.submit_segment(audio_np)
            # ────────────────────────────────────────────────

            self.on_speech_end(segment)

            self._current_start = None
            self._current_start_sample = None

        if self._total_samples % (self.sample_rate * 10) == 0:
            self._trim_buffer_lazy(keep_seconds=60)

    def _extract_segment_audio(self, start_sample: int, end_sample: int) -> Optional[torch.Tensor]:
        if start_sample < 0 or end_sample > self._total_samples or start_sample >= end_sample:
            return None

        target_len = end_sample - start_sample
        extracted = torch.zeros(target_len, dtype=torch.float32)

        with self._lock:
            current_pos = self._total_samples - sum(len(c) for c in self._audio_buffer)
            buffer_copy = list(self._audio_buffer)

        pos_in_buffer = 0
        out_idx = 0
        for chunk in buffer_copy:
            chunk_len = len(chunk)
            chunk_start = pos_in_buffer
            chunk_end = pos_in_buffer + chunk_len

            overlap_start = max(chunk_start, start_sample)
            overlap_end = min(chunk_end, end_sample)

            if overlap_start < overlap_end:
                rel_start = overlap_start - chunk_start
                rel_end = overlap_end - chunk_start
                seg = chunk[rel_start:rel_end]
                seg_len = len(seg)
                extracted[out_idx : out_idx + seg_len] = seg
                out_idx += seg_len

            pos_in_buffer += chunk_len
            if out_idx >= target_len:
                break

        return extracted if out_idx == target_len else None

    def _trim_buffer_lazy(self, keep_seconds: int = 60) -> None:
        with self._lock:
            max_samples = int(keep_seconds * self.sample_rate)
            samples_to_discard = max(0, self._total_samples - max_samples)
            old_total = self._total_samples
            old_chunks = len(self._audio_buffer)
            discarded = 0
            while self._audio_buffer and discarded < samples_to_discard:
                chunk = self._audio_buffer[0]
                if len(chunk) + discarded <= samples_to_discard:
                    discarded += len(chunk)
                    self._audio_buffer.popleft()
                else:
                    cut = samples_to_discard - discarded
                    self._audio_buffer[0] = chunk[cut:]
                    discarded += cut
                    break
            self._total_samples -= discarded

            if discarded > 0 or self.debug:
                current_sec = self._total_samples / self.sample_rate
                chunks = len(self._audio_buffer)
                if discarded > 0:
                    log.info(
                        f"[dim]Buffer trimmed:[/] −{discarded / self.sample_rate:.2f}s "
                        f"({discarded} samples) → "
                        f"[bold]{current_sec:.2f}s[/] kept "
                        f"({chunks} chunks)"
                    )
                elif self.debug and old_total % (self.sample_rate * 30) < (len(self._audio_buffer[0]) if self._audio_buffer else 1):
                    log.debug(
                        f"[dim]Buffer:[/] {current_sec:.2f}s "
                        f"({self._total_samples} samples, {chunks} chunks) – no trim needed"
                    )

    def _signal_handler(self, sig, frame):
        log.info("\nShutting down gracefully...")
        with self._lock:
            silent = torch.zeros(self.block_size)
            final = self.vad_iterator(silent, return_seconds=True)
            if final and "end" in final and self._current_start is not None:
                end_sec = final["end"]
                end_sample = int(end_sec * self.sample_rate)
                start_sample = self._current_start_sample
                duration_sec = end_sec - self._current_start
                segment = SpeechSegment(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_sec=self._current_start,
                    end_sec=end_sec,
                    duration_sec=duration_sec,
                )
                if self.save_segments:
                    audio_tensor = self._extract_segment_audio(start_sample, end_sample)
                    if audio_tensor is not None:
                        self._segment_counter += 1
                        seg_dir = Path(self.output_dir) / f"segment_{self._segment_counter:03d}"
                        seg_dir.mkdir(parents=True, exist_ok=True)

                        wav_path = seg_dir / "sound.wav"
                        json_path = seg_dir / "segment.json"

                        save_audio(str(wav_path), audio_tensor, self.sample_rate)

                        with json_path.open("w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "start_sample": segment.start_sample,
                                    "end_sample": segment.end_sample,
                                    "start_sec": segment.start_sec,
                                    "end_sec": segment.end_sec,
                                    "duration_sec": segment.duration_sec,
                                },
                                f,
                                indent=2,
                            )

                        log.info(f"[bold green]Saved final segment:[/] {seg_dir.name}")
                self.on_speech_end(segment)

    def start(self) -> None:
        log.info(f"Starting Silero VAD streamer • sr={self.sample_rate} • block={self.block_size}")
        log.info("Press Ctrl+C to stop.\n")
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=self.device,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        with self._stream:
            signal.signal(signal.SIGINT, self._signal_handler)
            try:
                while True:
                    sd.sleep(100)
            except KeyboardInterrupt:
                pass
        log.info("\nStream stopped.")

if __name__ == "__main__":
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    # Instantiate transcription pipeline probably already done at global, but can do here too if you want fresh instance.
    trans_pipeline = TranscriptionPipeline(max_workers=3)

    streamer = SileroVADStreamer(
        output_dir=OUTPUT_DIR,
        save_segments=True,
        debug=True,
    )
    streamer.start()