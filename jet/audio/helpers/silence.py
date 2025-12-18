import numpy as np
from threading import RLock
from typing import Optional
from jet.logger import logger
from jet.audio.utils import get_input_channels

SAMPLE_RATE = 16000
DTYPE = 'int16'

CHANNELS = min(2, get_input_channels())

# --- Module-level cache for calibrated silence threshold ---
_calibration_cache = {
    "threshold": None,
    "timestamp": None,
    "duration": None,
}
_cache_lock = RLock()


# def calibrate_silence_threshold(
#     calibration_duration: float = 2.0,
#     *,
#     force_recalibrate: bool = False,
#     max_age_seconds: float = 300
# ) -> float:
#     """
#     Calibrate silence threshold with smart caching.

#     Caches result for up to 5 minutes (or until force_recalibrate=True).
#     Thread-safe.
#     """
#     global _calibration_cache

#     with _cache_lock:
#         now = time.time()
#         cached = _calibration_cache["threshold"] is not None
#         too_old = (
#             _calibration_cache["timestamp"] is not None and
#             now - _calibration_cache["timestamp"] > max_age_seconds
#         )
#         same_duration = _calibration_cache["duration"] == calibration_duration

#         if not force_recalibrate and cached and same_duration and not too_old:
#             logger.info(
#                 f"Using cached silence threshold: {_calibration_cache['threshold']:.6f} "
#                 f"(age: {now - _calibration_cache['timestamp']:.1f}s)"
#             )
#             return _calibration_cache["threshold"]

#     # --- Actual calibration (only runs when needed) ---
#     logger.info(f"Calibrating silence threshold for {calibration_duration}s...")
#     calibration_frames = int(calibration_duration * SAMPLE_RATE)

#     stream = sd.InputStream(
#         samplerate=SAMPLE_RATE,
#         channels=CHANNELS,
#         dtype=DTYPE
#     )

#     with stream:
#         audio_data, _ = stream.read(calibration_frames)

#     # Compute median energy in RMS-like way (normalized to float)
#     audio_float = audio_data.astype(np.float32) / np.iinfo(DTYPE).max
#     energy_per_sample = np.square(audio_float)
#     median_energy = np.median(energy_per_sample)

#     # Threshold as a multiple of median - adjust 3x and minimum threshold
#     # realistic floor for float32 RMS-like energy
#     threshold = max(median_energy * 3.0, 0.005)

#     with _cache_lock:
#         _calibration_cache.update({
#             "threshold": threshold,
#             "timestamp": time.time(),
#             "duration": calibration_duration,
#         })

#     logger.info(
#         f"Calibration complete → threshold = {threshold:.6f} (cached)"
#     )
#     return threshold

def calibrate_silence_threshold(
    calibration_duration: float = 2.0,
    *,
    force_recalibrate: bool = False,
    max_age_seconds: float = 300
) -> float:
    return 0.01


def detect_silence(audio_chunk: np.ndarray, threshold: float) -> bool:
    """Detect if audio chunk is silent based on energy threshold."""
    from .energy import compute_energy
    energy = compute_energy(audio_chunk)
    is_silent = energy < threshold
    logger.debug(
        f"Audio chunk energy: {energy:.6f}, Threshold: {threshold:.6f}, Silent: {is_silent}")
    return is_silent


def trim_silent_chunks(audio_data: list[np.ndarray], threshold: Optional[float] = None) -> list[np.ndarray]:
    """Trim leading and trailing silent chunks from audio data list.

    Removes consecutive silent chunks only from the start and end of the sequence.
    Internal silent gaps between non-silent chunks are preserved.

    If threshold is None, it is obtained via calibrate_silence_threshold().
    """
    if threshold is None:
        threshold = calibrate_silence_threshold()

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
