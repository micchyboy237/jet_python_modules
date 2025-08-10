import sounddevice as sd
import numpy as np
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from jet.logger import logger


SAMPLE_RATE = 16000
DTYPE = 'int16'


def get_input_channels() -> int:
    device_info = sd.query_devices(sd.default.device[0], 'input')
    channels = device_info['max_input_channels']
    logger.debug(f"Detected {channels} input channels")
    return channels


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
    energy = np.mean(np.abs(audio_chunk))
    is_silent = energy < threshold
    logger.debug(
        f"Audio chunk energy: {energy:.6f}, Threshold: {threshold:.6f}, Silent: {is_silent}")
    return is_silent


def trim_silent_chunks(audio_data: list, threshold: float) -> list:
    """Trim silent chunks from start and end of audio data."""
    start_idx = 0
    end_idx = len(audio_data)

    # Trim from start
    for i, chunk in enumerate(audio_data):
        if not detect_silence(chunk, threshold):
            start_idx = i
            break

    # Trim from end
    for i in range(len(audio_data) - 1, -1, -1):
        if not detect_silence(audio_data[i], threshold):
            end_idx = i + 1
            break

    trimmed = audio_data[start_idx:end_idx]
    if start_idx > 0 or end_idx < len(audio_data):
        logger.info(
            f"Trimmed {start_idx} chunks from start, {len(audio_data) - end_idx} from end")
    return trimmed


def record_from_mic(
    duration: Optional[int] = None,
    silence_threshold: Optional[float] = None,
    silence_duration: float = 2.0
) -> Optional[np.ndarray]:
    """Record audio from microphone with silence detection and progress tracking."""
    silence_threshold = silence_threshold if silence_threshold is not None else calibrate_silence_threshold()

    duration_str = f"{duration}s" if duration is not None else "indefinite"
    logger.info(
        f"Starting recording: {CHANNELS} channel{'s' if CHANNELS > 1 else ''}, "
        f"max duration {duration_str}, silence threshold {silence_threshold:.6f}, "
        f"silence duration {silence_duration}s"
    )

    chunk_size = int(SAMPLE_RATE * 0.5)  # 0.5 second chunks
    max_frames = int(
        duration * SAMPLE_RATE) if duration is not None else float('inf')
    silence_frames = int(silence_duration * SAMPLE_RATE)
    grace_frames = int(SAMPLE_RATE * 1.0)  # 1-second grace period

    audio_data = []
    silent_count = 0
    recorded_frames = 0

    # Initialize progress bar: determinate if duration is set, indeterminate otherwise
    pbar_kwargs = {'total': duration, 'desc': "Recording", 'unit': "s",
                   'leave': True} if duration is not None else {'desc': "Recording", 'unit': "s", 'leave': True}
    with tqdm(**pbar_kwargs) as pbar:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE
        )

        with stream:
            while recorded_frames < max_frames:
                chunk = stream.read(chunk_size)[0]
                audio_data.append(chunk)
                recorded_frames += chunk_size
                pbar.update(0.5) if duration is not None else pbar.update(
                    0.5)  # Update by 0.5s

                # Skip silence detection during grace period
                if recorded_frames > grace_frames and detect_silence(chunk, silence_threshold):
                    silent_count += chunk_size
                    if silent_count >= silence_frames:
                        logger.info(
                            f"Silence detected for {silence_duration}s, stopping recording")
                        break
                else:
                    silent_count = 0

    if not audio_data:
        logger.warning("No audio recorded")
        return None

    # Trim silent chunks
    trimmed_data = trim_silent_chunks(audio_data, silence_threshold)
    if not trimmed_data:
        logger.warning("All chunks were silent after trimming")
        return None

    audio_data = np.concatenate(trimmed_data, axis=0)
    actual_duration = len(audio_data) / SAMPLE_RATE
    logger.info(f"Recording complete, actual duration: {actual_duration:.2f}s")
    return audio_data


def save_wav_file(filename, audio_data: np.ndarray):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(filename), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(np.dtype(DTYPE).itemsize)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    logger.info(f"Audio saved to {filename}")
