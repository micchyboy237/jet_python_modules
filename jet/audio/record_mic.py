import wave
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from jet.logger import logger
from jet.audio.helpers.silence import (
    SAMPLE_RATE,
    DTYPE,
    CHANNELS,
    calibrate_silence_threshold,
    detect_silence,
    trim_silent_chunks,
)


def record_from_mic(
    duration: Optional[int] = None,
    silence_threshold: Optional[float] = None,
    silence_duration: float = 2.0,
    trim_silent: bool = True,
) -> Optional[np.ndarray]:
    """Record audio from microphone with silence detection and progress tracking.

    Args:
        duration: Maximum recording duration in seconds (None = indefinite).
        silence_threshold: Silence level in RMS (None = auto-calibrated).
        silence_duration: Seconds of continuous silence to stop recording.
        trim_silent: If True, removes silent sections from the final audio.
    """
    silence_threshold = silence_threshold if silence_threshold is not None else calibrate_silence_threshold()

    duration_str = f"{duration}s" if duration is not None else "indefinite"
    logger.info(
        f"Starting recording: {CHANNELS} channel{'s' if CHANNELS > 1 else ''}, "
        f"max duration {duration_str}, silence threshold {silence_threshold:.6f}, "
        f"silence duration {silence_duration}s"
    )

    chunk_size = int(SAMPLE_RATE * 0.5)  # 0.5 second chunks
    max_frames = int(duration * SAMPLE_RATE) if duration is not None else float('inf')
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
                pbar.update(0.5) if duration is not None else pbar.update(0.5)  # Update by 0.5s

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

    # Trim silent chunks only when requested
    if trim_silent:
        trimmed_data = trim_silent_chunks(audio_data, silence_threshold)
        if not trimmed_data:
            logger.warning("All chunks were silent after trimming")
            return None
        audio_data = np.concatenate(trimmed_data, axis=0)
    else:
        # No trimming → just concatenate everything we recorded
        audio_data = np.concatenate(audio_data, axis=0)

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
