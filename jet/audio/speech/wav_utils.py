import io
import numpy as np
from typing import BinaryIO, Union
from pathlib import Path
import wave

from jet.audio.helpers.silence import (
    SAMPLE_RATE,
    DTYPE,
    CHANNELS,
)
from jet.logger import logger

def _validate_audio_data(audio_data: Union[np.ndarray, bytes]) -> bytes:
    sample_width = np.dtype(DTYPE).itemsize
    frame_size = CHANNELS * sample_width

    if isinstance(audio_data, np.ndarray):
        if not np.issubdtype(audio_data.dtype, np.integer):
            raise ValueError(f"Array must have integer dtype, got {audio_data.dtype}")
        if audio_data.size == 0:
            raise ValueError("Empty audio array")
        # Reshape to 2D if mono vector
        if audio_data.ndim == 1:
            if CHANNELS != 1:
                raise ValueError(f"Mono vector supplied but CHANNELS={CHANNELS}")
            audio_data = audio_data.reshape(-1, 1)
        elif audio_data.ndim != 2 or audio_data.shape[1] != CHANNELS:
            raise ValueError(f"Array must have shape (frames, {CHANNELS}) or (frames,) for mono")
        return audio_data.tobytes()

    # Raw bytes case
    if not isinstance(audio_data, (bytes, bytearray)):
        raise TypeError("audio_data must be np.ndarray or bytes/bytearray")
    if len(audio_data) == 0:
        raise ValueError("Empty audio bytes")
    if len(audio_data) % frame_size != 0:
        raise ValueError(
            f"Bytes length {len(audio_data)} not divisible by frame size {frame_size} "
            f"(channels={CHANNELS}, sample_width={sample_width})"
        )
    return bytes(audio_data)

def save_wav_file(filename, audio_data: Union[np.ndarray, bytes]) -> str:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    data_bytes = _validate_audio_data(audio_data)
    with wave.open(str(filename), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(np.dtype(DTYPE).itemsize)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data_bytes)
    abs_path = str(filename.resolve())
    return abs_path

def get_wav_bytes(audio_data: Union[np.ndarray, bytes]) -> bytes:
    """
    Generate WAV file bytes in memory without saving to disk.

    Accepts either np.ndarray or raw PCM bytes.
    """
    data_bytes = _validate_audio_data(audio_data)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(np.dtype(DTYPE).itemsize)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data_bytes)

    buffer.seek(0)
    wav_bytes = buffer.read()
    logger.info(f"Generated {len(wav_bytes)} bytes of in-memory WAV audio")
    return wav_bytes

def get_wav_fileobj(audio_data: Union[np.ndarray, bytes]) -> BinaryIO:
    """
    Generate a file-like object containing WAV data in memory.

    Accepts either np.ndarray or raw PCM bytes.
    """
    data_bytes = _validate_audio_data(audio_data)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(np.dtype(DTYPE).itemsize)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data_bytes)

    buffer.seek(0)
    logger.info("Generated in-memory WAV file-like object")
    return buffer
