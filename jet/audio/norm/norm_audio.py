# norm_audio.py

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
