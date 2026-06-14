import numpy as np
from jet.audio.helpers.config import SAMPLE_RATE


def get_audio_duration(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Return duration of audio in seconds.

    Parameters
    ----------
    samples : np.ndarray
        Input audio samples (1D array).

    Returns
    -------
    float
        Duration in seconds.
    """
    if len(samples) == 0:
        return 0.0

    return float(len(samples) / sample_rate)


def get_audio_buffer_duration(
    buffer: bytes | bytearray | None,
    sample_rate: int,  # ← still required, no default
) -> float:
    """
    Calculate duration of PCM audio buffer (assumes 16-bit integer samples).

    Currently hard-coded to 2 bytes per sample (int16).
    If you ever need to support other formats, add bytes_per_sample parameter back.
    """
    if not buffer:
        return 0.0

    BYTES_PER_SAMPLE = 2
    num_samples = len(buffer) // BYTES_PER_SAMPLE
    return num_samples / sample_rate
