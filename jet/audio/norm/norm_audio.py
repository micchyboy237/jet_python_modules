# norm_audio.py

from typing import Optional, Tuple

import librosa
import numpy as np


def normalize_audio(audio: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Peak-normalize *audio* to [-1.0, 1.0] only when it exceeds that range.

    This replicates the exact guard used in ``load_audio`` so that audio
    written with ``sf.write`` and then re-read produces an identical array —
    eliminating the amplitude shift that caused valley trough positions to
    differ between the live path and the saved-file path.

    Parameters
    ----------
    audio : np.ndarray
        1-D float32 array of audio samples.
    eps : float
        Tolerance above 1.0 that triggers normalisation (default: 1e-6).

    Returns
    -------
    np.ndarray
        Same array if already within [-1, 1+eps], otherwise peak-divided copy.
    """
    if len(audio) == 0:
        return audio
    peak = np.abs(audio).max()
    if peak > 1.0 + eps:
        return (audio / peak).astype(np.float32)
    return audio


def normalize_audio_for_vad(
    y: np.ndarray,
    sr: Optional[int] = None,
    method: str = "hybrid",  # Updated default
    target_rms_db: float = -20.0,  # Updated default
    max_peak: float = 0.95,  # Updated default
    eps: float = 1e-8,
) -> Tuple[np.ndarray, dict]:
    """
    Normalize audio specifically for Voice Activity Detection (VAD).

    Recommended for most pipelines: 'hybrid' method with target_rms_db=-20.
    This provides consistent levels for energy-based, WebRTC, Silero, and neural VADs.
    """

    if len(y) == 0:
        return y.astype(np.float32), {"method": method, "rms_db": -np.inf, "peak": 0.0}

    # Work on float32 copy
    y_norm = y.astype(np.float32).copy()

    # Original statistics
    original_rms = np.sqrt(np.mean(y_norm**2) + eps)
    original_peak = np.max(np.abs(y_norm))
    original_rms_db = 20 * np.log10(original_rms) if original_rms > eps else -np.inf

    if method == "peak":
        y_norm = librosa.util.normalize(y_norm, norm=np.inf)
        final_peak = 1.0

    elif method in ("rms", "hybrid"):
        target_rms = 10 ** (target_rms_db / 20.0)
        scale = target_rms / (original_rms + eps)
        y_norm *= scale

        current_peak = np.max(np.abs(y_norm))

        if method == "hybrid" and current_peak > max_peak:
            y_norm *= max_peak / (current_peak + eps)
            final_peak = max_peak
        else:
            final_peak = current_peak
    else:
        raise ValueError(f"Unknown method: {method}. Use 'peak', 'rms', or 'hybrid'.")

    # Final stats
    final_rms = np.sqrt(np.mean(y_norm**2) + eps)
    final_rms_db = 20 * np.log10(final_rms)

    info = {
        "method": method,
        "original_rms_db": round(original_rms_db, 2),
        "final_rms_db": round(final_rms_db, 2),
        "original_peak": round(float(original_peak), 4),
        "final_peak": round(float(final_peak), 4),
        "applied_gain_db": round(final_rms_db - original_rms_db, 2),
    }

    return y_norm, info


# ================ Example Usage ================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize audio for VAD.")
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    args = parser.parse_args()

    y, sr = librosa.load(args.audio_path, sr=None)

    y_norm, stats = normalize_audio_for_vad(y, sr=sr)

    print("Normalization applied:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
