from pathlib import Path
from typing import Dict, Any, List, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.ndimage import binary_opening, label

console = Console()


def _perform_simple_vad(
    rms: np.ndarray,
    zcr: np.ndarray,
    times: np.ndarray,
    sr: int,
    hop_length: int,
    rms_threshold: float = 0.015,
    zcr_threshold: float = 0.12,
    min_segment_frames: int = 15,
    opening_kernel: int = 5,
) -> List[Tuple[float, float]]:
    """
    Simple energy + ZCR based voice activity detection.
    Returns a list of (start_sec, end_sec) speech segments.
    """
    # Voiced speech typically has higher energy and lower ZCR
    speech_mask = (rms > rms_threshold) & (zcr < zcr_threshold)

    # Clean up small gaps/noise with morphological opening
    if opening_kernel > 1:
        speech_mask = binary_opening(speech_mask, structure=np.ones(opening_kernel))

    # Label connected components
    labeled, _ = label(speech_mask)

    segments: List[Tuple[float, float]] = []
    for component_id in range(1, labeled.max() + 1):
        component_times = times[labeled == component_id]
        if len(component_times) >= min_segment_frames:
            start = float(component_times[0])
            end = float(component_times[-1])
            segments.append((start, end))

    return segments


def extract_audio_insights_and_plots(
    audio_path: str | Path,
    output_dir: str | Path = "output_audio_analysis",
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    n_mfcc: int = 20,
    perform_vad: bool = True,
    vad_rms_threshold: float = 0.015,
    vad_zcr_threshold: float = 0.12,
) -> Dict[str, Any]:
    """
    Load an audio file, extract key features, generate insightful plots,
    and provide both global and segment-level (VAD) readable insights.

    Parameters
    ----------
    audio_path : str or Path
        Path to the audio file.
    output_dir : str or Path
        Directory to save plots (created if missing).
    n_fft, hop_length, n_mels, n_mfcc : int
        STFT / feature parameters.
    perform_vad : bool
        Whether to run simple voice activity detection for segment-level insights.
    vad_rms_threshold, vad_zcr_threshold : float
        Tunable thresholds for VAD.

    Returns
    -------
    insights : Dict[str, Any]
        Global statistics + VAD segment summary (if enabled).
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold green]Loading audio:[/bold green] {audio_path.name}")
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Frame-based features
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

    centroid = librosa.feature.spectral_centroid(S=S)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]

    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    # Global insights
    insights: Dict[str, Any] = {
        "duration_sec": round(duration, 2),
        "sample_rate_hz": int(sr),
        "mean_rms_energy": float(np.mean(rms)),
        "mean_zcr": float(np.mean(zcr)),
        "mean_spectral_centroid_hz": float(np.mean(centroid)),
        "mean_spectral_bandwidth_hz": float(np.mean(bandwidth)),
        "mean_spectral_rolloff_hz": float(np.mean(rolloff)),
        "mean_spectral_flatness": float(np.mean(flatness)),
    }

    # Voice Activity Detection (segment-level)
    segments: List[Tuple[float, float]] = []
    if perform_vad:
        segments = _perform_simple_vad(
            rms=rms,
            zcr=zcr,
            times=times,
            sr=sr,
            hop_length=hop_length,
            rms_threshold=vad_rms_threshold,
            zcr_threshold=vad_zcr_threshold,
        )
        total_speech = sum(end - start for start, end in segments)
        speech_pct = (total_speech / duration * 100) if duration > 0 else 0.0

        insights.update({
            "speech_percentage": round(speech_pct, 1),
            "num_speech_segments": len(segments),
            "total_speech_sec": round(total_speech, 1),
            "avg_segment_sec": round(total_speech / len(segments), 1) if segments else 0.0,
        })

    # Plotting
    plt.style.use("seaborn-v0_8-darkgrid")

    # 1. Waveform + RMS + ZCR
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set(title="Waveform")
    ax[1].plot(times, rms, color="orange")
    ax[1].set(title="RMS Energy", ylabel="Energy")
    ax[2].plot(times, zcr, color="purple")
    ax[2].set(title="Zero-Crossing Rate", xlabel="Time (s)")
    plt.tight_layout()
    fig.savefig(output_dir / "01_waveform_rms_zcr.png")
    plt.close(fig)

    # 2–6. Existing plots (unchanged)
    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(S_db, x_axis="time", y_axis="hz", sr=sr, hop_length=hop_length, ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="Power Spectrogram (dB)")
    fig.savefig(output_dir / "02_spectrogram.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(mel_db, x_axis="time", y_axis="mel", sr=sr, hop_length=hop_length, ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="Mel Spectrogram (dB)")
    fig.savefig(output_dir / "03_mel_spectrogram.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", sr=sr, hop_length=hop_length, ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title="Chromagram")
    fig.savefig(output_dir / "04_chromagram.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(mfcc, x_axis="time", sr=sr, hop_length=hop_length, ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title="MFCCs")
    fig.savefig(output_dir / "05_mfcc.png")
    plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(times, centroid, color="tab:blue")
    ax[0].set(title="Spectral Centroid", ylabel="Hz")
    ax[1].plot(times, bandwidth, color="tab:green")
    ax[1].set(title="Spectral Bandwidth", ylabel="Hz")
    ax[2].plot(times, rolloff, color="tab:red")
    ax[2].set(title="Spectral Roll-off", xlabel="Time (s)", ylabel="Hz")
    plt.tight_layout()
    fig.savefig(output_dir / "06_spectral_descriptors.png")
    plt.close(fig)

    # 7. New: Waveform with VAD overlay
    if perform_vad and segments:
        fig, ax = plt.subplots(figsize=(14, 4))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        for i, (start, end) in enumerate(segments):
            ax.axvspan(start, end, alpha=0.3, color="green",
                       label="Speech" if i == 0 else None)
        ax.set(title="Waveform with Voice Activity Detection (Speech Segments)")
        ax.legend()
        fig.savefig(output_dir / "07_waveform_vad.png")
        plt.close(fig)

    console.print(f"[bold green]All plots saved to:[/bold green] {output_dir.resolve()}")

    # Human-readable console summary
    console.print("\n[bold cyan]Global Audio Profile[/bold cyan]")
    table_global = Table(show_header=True, header_style="bold magenta")
    table_global.add_column("Aspect", justify="left")
    table_global.add_column("Interpretation", justify="left")
    table_global.add_column("Value", justify="right")

    loudness = "Quiet" if insights["mean_rms_energy"] < 0.05 else "Moderate" if insights["mean_rms_energy"] < 0.2 else "Loud"
    brightness = "Dark/Bass-heavy" if insights["mean_spectral_centroid_hz"] < 1200 else "Natural/Mid-range" if insights["mean_spectral_centroid_hz"] < 3000 else "Bright/Treble-rich"
    noisiness = "Very Tonal" if insights["mean_spectral_flatness"] < 0.05 else "Mostly Tonal" if insights["mean_spectral_flatness"] < 0.2 else "Somewhat Noisy"

    table_global.add_row("Duration", f"{insights['duration_sec']} seconds", "")
    table_global.add_row("Loudness", loudness, f"RMS {insights['mean_rms_energy']:.4f}")
    table_global.add_row("Brightness", brightness, f"Centroid {insights['mean_spectral_centroid_hz']:.0f} Hz")
    table_global.add_row("Noisiness", noisiness, f"Flatness {insights['mean_spectral_flatness']:.3f}")
    table_global.add_row("Voiced Content", "High" if insights["mean_zcr"] < 0.08 else "Moderate", f"ZCR {insights['mean_zcr']:.3f}")
    console.print(table_global)

    if perform_vad:
        console.print("\n[bold cyan]Segment-Level Insights (Voice Activity)[/bold cyan]")
        table_vad = Table(show_header=True, header_style="bold magenta")
        table_vad.add_column("Metric", justify="left")
        table_vad.add_column("Value", justify="right")
        table_vad.add_row("Active Speech", f"{insights['speech_percentage']}%")
        table_vad.add_row("Total Speech Time", f"{insights['total_speech_sec']} s")
        table_vad.add_row("Speech Segments", str(insights['num_speech_segments']))
        table_vad.add_row("Avg Segment Length", f"{insights['avg_segment_sec']} s")
        console.print(table_vad)

    return insights