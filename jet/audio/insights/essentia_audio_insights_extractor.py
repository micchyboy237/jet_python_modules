from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from numpy.typing import NDArray

import essentia.standard as ess

def load_audio(
    file_path: str | os.PathLike,
    sample_rate: int = 44100,
    mono: bool = True,
) -> NDArray[np.float32]:
    """
    Load an audio file using Essentia's MonoLoader.

    Parameters
    ----------
    file_path : str or PathLike
        Path to the audio file (supports common formats via FFmpeg).
    sample_rate : int, optional
        Target sample rate (default: 44100 Hz).
    mono : bool, optional
        Downmix to mono if True (default).

    Returns
    -------
    numpy.ndarray
        Audio signal as float32 array (shape: [samples,] for mono).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    loader = ess.MonoLoader(filename=str(file_path), sampleRate=sample_rate)
    audio: NDArray[np.float32] = loader()
    if not mono and audio.ndim == 1:
        # If stereo requested but file is mono, duplicate channel
        audio = np.stack([audio, audio], axis=0)
    return audio

def extract_loudness(
    audio: NDArray[np.float32],
    sample_rate: int = 44100,
) -> Dict[str, float]:
    """
    Extract loudness-related features (EBU R128 integrated, dynamic range).

    Returns
    -------
    dict
        Keys: 'integrated_loudness', 'loudness_range'.
    """
    loudness_ebu = ess.LoudnessEBUR128(sampleRate=sample_rate)
    # LoudnessEBUR128 requires stereo (channels x samples)
    # Broadcasting to (2, n) without relying on _to_stereo helper
    stereo_audio = np.repeat(audio[np.newaxis, :], 2, axis=0)
    integrated, _, _, _, range_loudness = loudness_ebu(stereo_audio)

    return {
        "integrated_loudness": float(integrated),
        "loudness_range": float(range_loudness),
    }

def extract_mfcc(
    audio: NDArray[np.float32],
    sample_rate: int = 44100,
    n_mfcc: int = 13,
) -> Dict[str, NDArray[np.float32]]:
    """
    Extract MFCCs (mean over frames).

    Returns
    -------
    dict
        Keys: 'mfcc_mean'.
    """
    spectrum = ess.Spectrum()
    window = ess.Windowing(type="hann")
    mfcc_alg = ess.MFCC(sampleRate=sample_rate, numberCoefficients=n_mfcc)

    mfccs: List[NDArray[np.float32]] = []
    for frame in ess.FrameGenerator(audio, frameSize=1024, hopSize=512):
        _, mfcc_coeffs = mfcc_alg(spectrum(window(frame)))
        mfccs.append(mfcc_coeffs)

    mfcc_array = np.array(mfccs)
    return {"mfcc_mean": mfcc_array.mean(axis=0)}

def extract_pitch_and_key(
    audio: NDArray[np.float32],
    sample_rate: int = 44100,
) -> Dict[str, Any]:
    """
    Extract predominant pitch salience and tonal key/scale.

    Returns
    -------
    dict
        Keys: 'pitch_salience_mean', 'key', 'scale', 'key_strength'.
    """
    # Pitch salience
    pitch_sal = ess.PitchSalience(sampleRate=sample_rate)
    salience_values = [pitch_sal(frame) for frame in ess.FrameGenerator(audio, frameSize=2048, hopSize=512)]
    mean_salience = float(np.mean(salience_values))

    # Key/scale
    key_alg = ess.Key()
    # Returns tuple of 4: key, scale, strength, first_to_second_relative_strength
    key, scale, strength, _ = key_alg(audio)

    return {
        "pitch_salience_mean": mean_salience,
        "key": key,
        "scale": scale,
        "key_strength": float(strength),
    }

def extract_rhythm(
    audio: NDArray[np.float32],
    sample_rate: int = 44100,
) -> Dict[str, Any]:
    """
    Extract BPM and beat positions.

    Returns
    -------
    dict
        Keys: 'bpm', 'beats_positions' (list of seconds).
    """
    rhythm_extractor = ess.RhythmExtractor2013()
    # Returns: bpm, beats_positions, confidence, beats_intervals, _ (5 values)
    bpm, beats, _, _, _ = rhythm_extractor(audio)

    return {
        "bpm": float(bpm),
        "beats_positions": beats.tolist(),
    }

def extract_all_insights(
    file_path: str | os.PathLike,
    sample_rate: int = 44100,
) -> Dict[str, Any]:
    """
    High-level function to extract a comprehensive set of audio insights.

    Parameters
    ----------
    file_path : str or PathLike
        Audio file path.
    sample_rate : int, optional
        Resample rate.

    Returns
    -------
    dict
        Nested dictionary with all extracted features.
    """
    audio = load_audio(file_path, sample_rate=sample_rate)

    insights: Dict[str, Any] = {
        "loudness": extract_loudness(audio, sample_rate),
        "spectral": extract_mfcc(audio, sample_rate),
        "tonal": extract_pitch_and_key(audio, sample_rate),
        "rhythm": extract_rhythm(audio, sample_rate),
    }

    return insights

if __name__ == "__main__":
    insights = extract_all_insights("path/to/your/audio.wav")
    print(insights["rhythm"]["bpm"])
    print(insights["tonal"]["key"])
