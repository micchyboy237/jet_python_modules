from threading import RLock
from typing import Optional

import numpy as np
from fireredvad.core.constants import (
    FRAME_LENGTH_SAMPLE,
    FRAME_SHIFT_SAMPLE,
    SAMPLE_RATE,
)
from jet.audio.utils import get_input_channels
from jet.logger import logger

DTYPE = "int16"
CHANNELS = min(2, get_input_channels())

_calibration_cache = {
    "threshold": None,
    "timestamp": None,
    "duration": None,
}
_cache_lock = RLock()


def calibrate_silence_threshold(
    calibration_duration: float = 2.0,
    *,
    force_recalibrate: bool = False,
    max_age_seconds: float = 300,
) -> float:
    return 0.01


def detect_silence(
    audio_chunk: np.ndarray, threshold: float = 0.01, verbose: bool = False
) -> bool:
    """Detect if audio chunk is silent based on energy threshold."""
    from .energy import compute_energy

    energy = compute_energy(audio_chunk)
    is_silent = energy < threshold
    if verbose:
        logger.debug(
            f"Audio chunk energy: {energy:.6f}, Threshold: {threshold:.6f}, Silent: {is_silent}"
        )
    return is_silent


def trim_silent_chunks(
    audio_data: list[np.ndarray], threshold: Optional[float] = None
) -> list[np.ndarray]:
    """Trim leading and trailing silent chunks from audio data list.

    Removes consecutive silent chunks only from the start and end of the sequence.
    Internal silent gaps between non-silent chunks are preserved.

    If threshold is None, it is obtained via calibrate_silence_threshold().
    """
    if threshold is None:
        threshold = calibrate_silence_threshold()

    start_idx = 0
    end_idx = len(audio_data)

    for i, chunk in enumerate(audio_data):
        if not detect_silence(chunk, threshold):
            start_idx = i
            break
    else:
        return []

    for i in range(len(audio_data) - 1, -1, -1):
        if not detect_silence(audio_data[i], threshold):
            end_idx = i + 1
            break

    trimmed = audio_data[start_idx:end_idx]

    if start_idx > 0 or end_idx < len(audio_data):
        logger.info(
            f"Trimmed {start_idx} chunks from start, {len(audio_data) - end_idx} from end"
        )

    return trimmed


def get_latest_true_silent_duration(
    audio_chunks: list[np.ndarray],
    threshold: Optional[float] = None,
    sample_rate: int = SAMPLE_RATE,
) -> float:
    """Get the duration in seconds of the latest consecutive true silent frames.

    Scans from the end of the audio_chunks list backwards and counts how many
    trailing chunks are all silent. Returns the total duration in seconds.

    Args:
        audio_chunks: List of audio chunk arrays.
        threshold: Energy threshold for silence detection. Defaults to calibrated value.
        sample_rate: Sample rate of the audio. Defaults to firered SAMPLE_RATE.

    Returns:
        Duration in seconds of the latest trailing silent frames. 0.0 if none.
    """
    if threshold is None:
        threshold = calibrate_silence_threshold()

    silent_samples = 0
    for chunk in reversed(audio_chunks):
        if detect_silence(chunk, threshold):
            silent_samples += len(chunk)
        else:
            break

    return silent_samples / sample_rate


def has_true_silence_in_latest_frames(
    audio_chunk: np.ndarray,
    num_latest_frames: int,
    threshold: Optional[float] = None,
    frame_length: int = None,
    hop_size: int = None,
) -> bool:
    """Check if the input audio_chunk contains any true silent frames in its latest frames.

    Splits the audio chunk into overlapping frames using the configured frame/hop sizes,
    then checks only the last `num_latest_frames` frames for silence.

    Args:
        audio_chunk: 1-D numpy array of audio samples.
        num_latest_frames: How many trailing frames to inspect.
        threshold: Energy threshold for silence detection. Defaults to calibrated value.
        frame_length: Samples per frame. Defaults to firered FRAME_LENGTH_SAMPLE (400).
        hop_size: Hop size in samples. Defaults to firered FRAME_SHIFT_SAMPLE (160).

    Returns:
        True if at least one of the latest frames is silent, False otherwise.
    """
    if frame_length is None:
        frame_length = FRAME_LENGTH_SAMPLE  # 400 samples (25 ms)
    if hop_size is None:
        hop_size = FRAME_SHIFT_SAMPLE  # 160 samples (10 ms)
    if threshold is None:
        threshold = calibrate_silence_threshold()

    frames: list[np.ndarray] = []
    start = 0
    while start + frame_length <= len(audio_chunk):
        frames.append(audio_chunk[start : start + frame_length])
        start += hop_size

    if not frames:
        frames = [audio_chunk]

    latest_frames = frames[-num_latest_frames:]
    return any(detect_silence(frame, threshold) for frame in latest_frames)
