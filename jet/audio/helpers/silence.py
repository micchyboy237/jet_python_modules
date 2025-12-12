import numpy as np
import sounddevice as sd

from jet.logger import logger
from jet.audio.utils import get_input_channels


SAMPLE_RATE = 16000
DTYPE = 'int16'

CHANNELS = min(2, get_input_channels())


def calibrate_silence_threshold(calibration_duration: float = 2.0) -> float:
    """Calibrate silence threshold based on ambient noise using median energy."""
    logger.info(
        f"Calibrating silence threshold for {calibration_duration}s...")
    calibration_frames = int(calibration_duration * SAMPLE_RATE)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE
    )

    with stream:
        audio_data = stream.read(calibration_frames)[0]

    energies = np.abs(audio_data)
    avg_energy = np.median(energies)  # Use median to reduce impact of outliers
    min_energy = np.min(energies)
    max_energy = np.max(energies)

    # Set threshold as 1.5x the median noise energy, with a minimum
    threshold = max(avg_energy * 1.5, 0.01)
    logger.info(
        f"Calibration complete. Median energy: {avg_energy:.6f}, "
        f"Min energy: {min_energy:.6f}, Max energy: {max_energy:.6f}, "
        f"Set threshold: {threshold:.6f}"
    )
    return threshold


def detect_silence(audio_chunk: np.ndarray, threshold: float) -> bool:
    """Detect if audio chunk is silent based on energy threshold."""
    from .energy import compute_energy
    energy = compute_energy(audio_chunk)
    is_silent = energy < threshold
    logger.debug(
        f"Audio chunk energy: {energy:.6f}, Threshold: {threshold:.6f}, Silent: {is_silent}")
    return is_silent


def trim_silent_chunks(audio_data: list, threshold: float) -> list:
    """Trim silent chunks from start and end of audio data."""
    start_idx = 0
    end_idx = len(audio_data)

    # Find first non-silent chunk
    for i, chunk in enumerate(audio_data):
        if not detect_silence(chunk, threshold):
            start_idx = i
            break
    else:
        # All chunks are silent
        return []

    # Find last non-silent chunk
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
