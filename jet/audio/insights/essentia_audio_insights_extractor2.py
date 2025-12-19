from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

import essentia.standard as es

console = Console()


def extract_essentia_insights_and_plots(
    audio_path: str | Path,
    output_dir: str | Path = "output_essentia_analysis",
    lowlevel_stats: List[str] = None,
    rhythm_stats: List[str] = None,
    tonal_stats: List[str] = None,
    compute_highlevel: bool = False,
    profile: str | None = None,
) -> Dict[str, Any]:
    """
    Extract comprehensive audio insights using Essentia's MusicExtractor.

    This leverages Essentia's powerful MusicExtractor to compute a vast set of
    low-level, rhythm, tonal, and optional high-level descriptors.

    Features include:
    - Low-level: spectral (centroid, rolloff, etc.), energy, ZCR, silence, etc.
    - Rhythm: BPM, beat positions, onset rate, danceability.
    - Tonal: key/scale, chords, tuning frequency, HPCP chroma.
    - High-level (optional): genre, mood, voice/instrumental probabilities.

    Generates insightful plots (waveform, spectrogram, key/chords, beats).

    Parameters
    ----------
    audio_path : str or Path
        Path to the audio file.
    output_dir : str or Path
        Directory to save plots.
    lowlevel_stats, rhythm_stats, tonal_stats : List[str], optional
        Statistics to aggregate (e.g., ['mean', 'var', 'median']). Defaults to common set.
    compute_highlevel : bool
        If True, compute high-level classifiers (requires models).
    profile : str | None
        Path to custom YAML profile for MusicExtractor (advanced tuning).

    Returns
    -------
    insights : Dict[str, Any]
        Nested dictionary of all extracted features (Essentia Pool format).
    """
    if lowlevel_stats is None:
        lowlevel_stats = ["mean", "var", "median", "min", "max"]
    if rhythm_stats is None:
        rhythm_stats = ["mean", "var"]
    if tonal_stats is None:
        tonal_stats = ["mean", "var"]

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold green]Loading and analyzing audio with Essentia:[/bold green] {audio_path.name}")

    # Instantiate MusicExtractor with configurable stats
    extractor = es.MusicExtractor(
        lowlevelStats=lowlevel_stats,
        rhythmStats=rhythm_stats,
        tonalStats=tonal_stats,
        highlevelStats=["mean"] if compute_highlevel else [],
        profile=profile or "",  # Empty string uses default
    )

    # Run extraction: returns aggregated features + frame-wise features
    features, features_frames = extractor(str(audio_path))

    # Basic metadata
    duration = features["metadata.audio_properties.length"]
    sr = features["metadata.audio_properties.sample_rate"]
    bpm = features["rhythm.bpm"]
    key = f"{features['tonal.key_key']} {features['tonal.key_scale']}"
    danceability = features["highlevel.danceability.mean"] if compute_highlevel else None

    # Global insights summary
    insights_summary = {
        "duration_sec": round(duration, 2),
        "sample_rate_hz": int(sr),
        "estimated_bpm": round(bpm, 1),
        "estimated_key": key,
        "danceability": round(danceability, 3) if danceability else "N/A",
        "mean_spectral_centroid_hz": round(features["lowlevel.spectral_centroid.mean"], 1),
        "mean_rms_energy": round(features["lowlevel.average_loudness"], 4),
        "onset_rate": round(features["rhythm.onset_rate"], 2),
    }

    # Plotting (using frame-wise data where available)
    plt.style.use("seaborn-v0_8-darkgrid")

    # 1. Waveform (load audio separately for plot)
    audio = es.MonoLoader(filename=str(audio_path), sampleRate=sr)()
    times = np.linspace(0, duration, len(audio))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, audio)
    ax.set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")
    fig.savefig(output_dir / "01_waveform.png")
    plt.close(fig)

    # 2. Mel Spectrogram (recompute for visualization)
    mel = es.MelBands(numberBands=128)(es.Spectrum()(es.Windowing()(es.FrameGenerator(audio, frameSize=2048, hopSize=512))))
    mel_db = es.PowerToDb()(mel)
    fig, ax = plt.subplots(figsize=(12, 6))
    img = ax.imshow(mel_db.T, origin="lower", aspect="auto", interpolation="nearest")
    ax.set(title="Mel Spectrogram (dB)", xlabel="Frames", ylabel="Mel Bands")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.savefig(output_dir / "02_mel_spectrogram.png")
    plt.close(fig)

    # 3. Beats overlay on waveform
    beats = features["rhythm.beats_position"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, audio, alpha=0.6)
    for beat in beats:
        ax.axvline(beat, color="red", alpha=0.7)
    ax.set(title=f"Waveform with Beats (BPM: {bpm:.1f})", xlabel="Time (s)")
    fig.savefig(output_dir / "03_waveform_beats.png")
    plt.close(fig)

    # 4. Key/Chords (HPCP chroma if available)
    if "tonal.hpcp" in features_frames.descriptorNames():
        hpcp = np.array(features_frames["tonal.hpcp"])
        fig, ax = plt.subplots(figsize=(12, 6))
        img = ax.imshow(hpcp.T, aspect="auto", origin="lower", interpolation="nearest")
        ax.set(title="HPCP Chromagram", xlabel="Frames", ylabel="Pitch Class")
        fig.colorbar(img, ax=ax)
        fig.savefig(output_dir / "04_chromagram.png")
        plt.close(fig)

    console.print(f"[bold green]Analysis complete. Plots saved to:[/bold green] {output_dir.resolve()}")

    # Human-readable summary tables
    console.print("\n[bold cyan]Global Audio Insights (Essentia)[/bold cyan]")
    table_global = Table()
    table_global.add_column("Aspect", justify="left")
    table_global.add_column("Value", justify="right")
    for k, v in insights_summary.items():
        table_global.add_row(k.replace("_", " ").title(), str(v))
    console.print(table_global)

    if compute_highlevel:
        console.print("\n[bold cyan]High-Level Classifications[/bold cyan]")
        table_hl = Table()
        table_hl.add_column("Descriptor", justify="left")
        table_hl.add_column("Probability", justify="right")
        for desc in features.descriptorNames():
            if desc.startswith("highlevel.") and desc.endswith(".mean"):
                name = desc.replace("highlevel.", "").replace(".mean", "").replace("_", " ").title()
                table_hl.add_row(name, f"{features[desc]:.3f}")
        console.print(table_hl)

    # Save full features to JSON for further use
    json_path = output_dir / "full_features.json"
    es.YamlOutput(filename=str(json_path), format="json")(features)
    console.print(f"[bold blue]Full feature set saved to:[/bold blue] {json_path}")

    return features  # Return the full Pool for advanced use


if __name__ == "__main__":
    audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/full_recording.wav"
    insights = extract_essentia_insights_and_plots(
        audio_path,
        compute_highlevel=False,  # Set True if you have models
    )
