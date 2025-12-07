# audio_preprocessor.py
from __future__ import annotations

import io
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps
from rich.console import Console

console = Console()

SampleRate = Literal[16000]
Channels = Literal[1]

# --- SINGLETONS for model and resampler(s) ---
_VAD_MODEL: torch.nn.Module | None = None
_RESAMPLERS: dict[tuple[int, int], torchaudio.transforms.Resample] = {}


def _get_vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        console.log("[bold green]Loading Silero VAD model (singleton)...[/bold green]")
        _VAD_MODEL = load_silero_vad()
        console.log("[bold green]Silero VAD model loaded[/bold green]")
    return _VAD_MODEL


def _get_resampler(orig_sr: int, target_sr: int):
    key = (orig_sr, target_sr)
    if key not in _RESAMPLERS:
        _RESAMPLERS[key] = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr,
            resampling_method="kaiser_window",
        )
    return _RESAMPLERS[key]


class PreprocessResult(TypedDict):
    audio: np.ndarray
    sample_rate: int
    duration_sec: float
    original_duration_sec: float
    vad_kept_ratio: float


class AudioPreprocessor:
    def __init__(
        self,
        target_sr: SampleRate = 16000,
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        padding_duration: float = 0.1,
    ):
        self.target_sr = target_sr
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.padding_duration = padding_duration

        self.model = _get_vad_model()

    def load_audio(self, file_path: str | Path | bytes) -> tuple[np.ndarray, int]:
        # Use torchaudio for robust, fast, format-agnostic loading!
        if isinstance(file_path, bytes):
            info = torchaudio.info(io.BytesIO(file_path))
            waveform, sr = torchaudio.load(io.BytesIO(file_path))
        else:
            waveform, sr = torchaudio.load(file_path)

        audio = waveform.numpy().mean(axis=0)  # to mono
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        console.log(f"Loaded audio: sr={sr}Hz, duration={len(audio)/sr:.2f}s, peak={np.abs(audio).max():.4f}")
        return audio, sr

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.target_sr:
            console.log("No resampling needed (already 16kHz)")
            return audio

        resampler = _get_resampler(orig_sr, self.target_sr)
        tensor = torch.from_numpy(audio).unsqueeze(0)
        resampled = resampler(tensor).squeeze(0).numpy()
        console.log(f"Resampled {orig_sr} → {self.target_sr}Hz, new length: {len(resampled)}")
        return resampled

    def normalize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """Normalize to level Silero VAD expects (~ -6 dBFS → peak ≈ 0.5)"""
        peak = np.max(np.abs(audio))
        if peak == 0:
            console.log("[yellow]Silent audio, skipping normalization[/yellow]")
            return audio

        target_peak = 0.5          # ← THIS IS WHAT SILERO WANTS
        audio = audio * (target_peak / peak)
        audio = np.clip(audio, -1.0, 1.0)
        console.log(f"Normalization: {peak:.4f} → {np.max(np.abs(audio)):.4f} (target 0.5)")
        return audio


    def apply_vad(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != self.target_sr:
            raise ValueError("VAD requires 16kHz audio")

        peak = np.max(np.abs(audio))
        console.log(f"[VAD] Input peak: {peak:.4f}, samples: {len(audio)}")

        # Only boost if audio is unnaturally quiet (shouldn't happen after proper norm)
        if peak < 0.2:
            boost = 0.5 / peak
            audio = audio * boost
            console.log(f"[VAD] Boosted low-energy audio x{boost:.2f}")

        audio_tensor = torch.from_numpy(audio).float()

        timestamps = get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=int(self.min_speech_duration * 1000),
            min_silence_duration_ms=100,
            speech_pad_ms=int(self.padding_duration * 1000),
            return_seconds=False,
        )

        console.log(f"[VAD] Detected {len(timestamps)} speech segments → {timestamps}")

        if not timestamps:
            console.log("[bold yellow]No speech detected → fallback to full[/bold yellow]")
            return audio

        segments = [audio[int(ts["start"]):int(ts["end"])] for ts in timestamps]
        vad_audio = np.concatenate(segments)
        console.log(f"[VAD] Final VAD output: {len(vad_audio)/sample_rate:.2f}s")
        return vad_audio

    def preprocess(
        self, file_path: str | Path | bytes, apply_vad: bool = False
    ) -> PreprocessResult:
        original_audio, orig_sr = self.load_audio(file_path)
        original_duration = len(original_audio) / orig_sr
        console.log(f"Original duration: {original_duration:.2f}s")

        audio = self.resample(original_audio, orig_sr)
        audio = self.normalize_loudness(audio)

        vad_kept_ratio = 1.0
        if apply_vad:
            console.log("[bold blue]Applying VAD...[/bold blue]")
            audio = self.apply_vad(audio, self.target_sr)
            vad_kept_ratio = len(audio) / (original_duration * self.target_sr)
            console.log(f"VAD kept ratio: {vad_kept_ratio:.3f}")

        duration_sec = len(audio) / self.target_sr

        console.log(f"[bold green]Final output: {duration_sec:.2f}s, kept {vad_kept_ratio:.1%} of original[/bold green]")

        return {
            "audio": audio,
            "sample_rate": self.target_sr,
            "duration_sec": duration_sec,
            "original_duration_sec": original_duration,
            "vad_kept_ratio": vad_kept_ratio,
        }
