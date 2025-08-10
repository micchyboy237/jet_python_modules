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


def detect_silence(audio_chunk: np.ndarray, threshold: float = 0.01) -> bool:
    """Detect if audio chunk is silent based on energy threshold."""
    energy = np.mean(np.abs(audio_chunk))
    is_silent = energy < threshold
    logger.debug(f"Audio chunk energy: {energy:.6f}, Silent: {is_silent}")
    return is_silent


def record_from_mic(
    duration: int,
    silence_threshold: float = 0.01,
    silence_duration: float = 2.0
) -> Optional[np.ndarray]:
    """Record audio from microphone with silence detection and progress tracking."""
    logger.info(
        f"Starting recording: {CHANNELS} channel{'s' if CHANNELS > 1 else ''}, "
        f"max duration {duration}s, silence threshold {silence_threshold}, "
        f"silence duration {silence_duration}s"
    )

    chunk_size = int(SAMPLE_RATE * 0.5)  # 0.5 second chunks
    max_frames = int(duration * SAMPLE_RATE)
    silence_frames = int(silence_duration * SAMPLE_RATE)

    audio_data = []
    silent_count = 0
    recorded_frames = 0

    # Initialize progress bar
    with tqdm(total=duration, desc="Recording", unit="s", leave=True) as pbar:
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
                pbar.update(0.5)  # Update progress bar by 0.5 seconds

                if detect_silence(chunk, silence_threshold):
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

    audio_data = np.concatenate(audio_data, axis=0)
    actual_duration = recorded_frames / SAMPLE_RATE
    logger.info(f"Recording complete, actual duration: {actual_duration:.2f}s")
    return audio_data[:recorded_frames]


def save_wav_file(filename, audio_data: np.ndarray):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(filename), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(np.dtype(DTYPE).itemsize)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    logger.info(f"Audio saved to {filename}")
