import sounddevice as sd
import numpy as np
import asyncio
from typing import Generator, AsyncGenerator, Optional, Tuple, Dict
from tqdm import tqdm
from jet.audio.helpers.silence import (
    SAMPLE_RATE,
    DTYPE,
    CHANNELS,
    calibrate_silence_threshold,
    detect_silence,
)
from jet.audio.speech.wav_utils import save_wav_file
from jet.logger import logger


def trim_silent_portions(
    chunk: np.ndarray,
    silence_threshold: float,
    sub_chunk_duration: float = 0.1
) -> Tuple[Optional[np.ndarray], int, int]:
    """Trim silent portions from the start and end of an audio chunk."""
    sub_chunk_size = int(SAMPLE_RATE * sub_chunk_duration)
    if chunk.shape[0] < sub_chunk_size:
        logger.debug(f"Chunk too small for trimming: {chunk.shape[0]} samples")
        return (chunk, 0, chunk.shape[0]) if not detect_silence(chunk, silence_threshold) else (None, 0, 0)

    sub_chunks = [chunk[i:i + sub_chunk_size] for i in range(0, chunk.shape[0], sub_chunk_size)]
    start_idx = 0
    end_idx = len(chunk)

    for i, sub_chunk in enumerate(sub_chunks):
        if len(sub_chunk) >= sub_chunk_size and not detect_silence(sub_chunk, silence_threshold):
            start_idx = i * sub_chunk_size
            break
    else:
        return None, 0, 0

    for i in range(len(sub_chunks) - 1, -1, -1):
        if len(sub_chunks[i]) >= sub_chunk_size and not detect_silence(sub_chunks[i], silence_threshold):
            end_idx = (i + 1) * sub_chunk_size
            break

    trimmed_chunk = chunk[start_idx:end_idx]
    logger.debug(f"Trimmed {start_idx} from start, {chunk.shape[0] - end_idx} from end")
    return trimmed_chunk, start_idx, end_idx


def save_chunk(
    chunk: np.ndarray,
    chunk_index: int,
    cumulative_duration: float,
    silence_threshold: float,
    overlap_samples: int,
    output_dir: str
) -> Tuple[Optional[str], Optional[Dict]]:
    """Save trimmed chunk with overlap preservation."""
    start_overlap = overlap_samples if chunk_index > 0 else 0
    end_overlap = overlap_samples
    non_overlap_start = start_overlap
    non_overlap_end = len(chunk) - end_overlap

    if non_overlap_end <= non_overlap_start:
        # Handle edge case: only overlap exists
        start_overlap_chunk = chunk[:start_overlap] if start_overlap > 0 else np.array([], dtype=chunk.dtype).reshape(0, CHANNELS)
        end_overlap_chunk = chunk[-end_overlap:] if end_overlap > 0 else np.array([], dtype=chunk.dtype).reshape(0, CHANNELS)
        if np.mean(np.abs(start_overlap_chunk)) > silence_threshold or np.mean(np.abs(end_overlap_chunk)) > silence_threshold:
            final_chunk = chunk
        else:
            return None, None
    else:
        start_overlap_chunk = chunk[:start_overlap] if start_overlap > 0 else np.array([], dtype=chunk.dtype).reshape(0, CHANNELS)
        non_overlap_chunk = chunk[non_overlap_start:non_overlap_end]
        end_overlap_chunk = chunk[-end_overlap:] if end_overlap > 0 else np.array([], dtype=chunk.dtype).reshape(0, CHANNELS)

        trimmed_non_overlap, trim_start_idx, trim_end_idx = trim_silent_portions(non_overlap_chunk, silence_threshold)
        if trimmed_non_overlap is None or len(trimmed_non_overlap) == 0:
            if np.mean(np.abs(start_overlap_chunk)) > silence_threshold or np.mean(np.abs(end_overlap_chunk)) > silence_threshold:
                final_chunk = chunk
            else:
                return None, None
        else:
            chunks_to_concat = [c for c in [start_overlap_chunk, trimmed_non_overlap, end_overlap_chunk] if len(c) > 0]
            final_chunk = np.concatenate(chunks_to_concat, axis=0)

    chunk_filename = f"{output_dir}/stream_chunk_{chunk_index:04d}.wav"
    save_wav_file(chunk_filename, final_chunk)
    chunk_duration = len(final_chunk) / SAMPLE_RATE

    metadata = {
        "chunk_index": chunk_index,
        "filename": chunk_filename,
        "duration_s": round(chunk_duration, 3),
        "sample_count": len(final_chunk),
        "start_time_s": round(cumulative_duration, 3),
        "end_time_s": round(cumulative_duration + chunk_duration, 3),
        "start_overlap_samples": start_overlap,
        "end_overlap_samples": end_overlap
    }
    return chunk_filename, metadata


# ————————————————————————————————————————————————————————
# 1. Original sync generator (kept for backward compatibility)
# ————————————————————————————————————————————————————————
def stream_non_silent_audio(
    silence_threshold: Optional[float] = None,
    chunk_duration: float = 0.5,
    silence_duration: float = 2.0,
    min_chunk_duration: float = 1.0,
    overlap_duration: float = 0.0
) -> Generator[np.ndarray, None, None]:
    silence_threshold = silence_threshold or calibrate_silence_threshold()
    chunk_size = int(SAMPLE_RATE * chunk_duration)
    silence_frames = int(silence_duration * SAMPLE_RATE)
    min_chunk_samples = int(SAMPLE_RATE * min_chunk_duration)
    overlap_samples = int(SAMPLE_RATE * overlap_duration)

    silent_count = 0
    buffer = []
    buffer_samples = 0
    overlap_buffer = np.zeros((0, CHANNELS), dtype=DTYPE)

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, blocksize=chunk_size)
    stream.start()

    try:
        with tqdm(desc="Streaming chunks (sync)", unit="chunk", leave=True) as pbar:
            while True:
                chunk, overflowed = stream.read(chunk_size)
                if overflowed:
                    logger.warning("Buffer overflow!")
                buffer.append(chunk)
                buffer_samples += chunk.shape[0]

                if not detect_silence(chunk, silence_threshold):
                    silent_count = 0
                    if buffer_samples >= min_chunk_samples + overlap_samples:
                        yield_chunk = np.concatenate([overlap_buffer, *buffer], axis=0)
                        yield yield_chunk
                        overlap_buffer = yield_chunk[-overlap_samples:] if overlap_samples > 0 else np.zeros((0, CHANNELS), dtype=DTYPE)
                        buffer = []
                        buffer_samples = 0
                        pbar.update(1)
                else:
                    silent_count += chunk_size
                    if silent_count >= silence_frames and buffer_samples >= min_chunk_samples:
                        final_chunk = np.concatenate([overlap_buffer, *buffer], axis=0)
                        if not detect_silence(final_chunk, silence_threshold):
                            yield final_chunk
                        break
    finally:
        stream.stop()
        stream.close()


# ————————————————————————————————————————————————————————
# 2. NEW: Async generator — enables real parallel transcription
# ————————————————————————————————————————————————————————
async def async_stream_non_silent_audio(
    silence_threshold: Optional[float] = None,
    chunk_duration: float = 0.5,
    silence_duration: float = 5.0,
    min_chunk_duration: float = 5.0,
    overlap_duration: float = 2.0,
    auto_close_on_long_silence: bool = False,
) -> AsyncGenerator[np.ndarray, None]:
    """
    Fully async version — yields non-silent audio chunks with overlap support.
    Designed for real-time transcription pipelines where transcription runs in parallel.

    Parameters
    ----------
    silence_threshold : float, optional
        Energy threshold to consider a chunk silent. Auto-calibrated if None.
    chunk_duration : float, default 0.5
        Duration of each raw audio block read from microphone (seconds).
    silence_duration : float, default 5.0
        How long continuous silence must last to trigger auto-close (if enabled).
    min_chunk_duration : float, default 5.0
        Minimum accumulated non-silent audio required before yielding a chunk.
    overlap_duration : float, default 2.0
        Overlap between consecutive yielded chunks (for context in transcription).
    auto_close_on_long_silence : bool, default False
        If True → stream automatically ends after `silence_duration` seconds of silence
        (once at least one valid chunk has been yielded).
        If False → long silence is ignored; stream continues indefinitely until manual stop (Ctrl+C).

    Yields
    ------
    np.ndarray
        Audio chunk (int16, shape: [samples, channels]) ready for saving/transcription.
    """
    silence_threshold = silence_threshold or calibrate_silence_threshold()
    chunk_size = int(SAMPLE_RATE * chunk_duration)
    silence_frames = int(silence_duration * SAMPLE_RATE)
    min_chunk_samples = int(SAMPLE_RATE * min_chunk_duration)
    overlap_samples = int(SAMPLE_RATE * overlap_duration)

    silent_count = 0
    buffer: list[np.ndarray] = []
    buffer_samples = 0
    overlap_buffer = np.zeros((0, CHANNELS), dtype=DTYPE)

    logger.info(
        f"Starting ASYNC audio stream | "
        f"overlap={overlap_duration}s, min_chunk={min_chunk_duration}s, "
        f"auto_close_on_long_silence={auto_close_on_long_silence}"
    )

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=chunk_size,
    )
    stream.start()

    try:
        while True:
            chunk, overflowed = await asyncio.get_running_loop().run_in_executor(
                None, stream.read, chunk_size
            )

            if overflowed:
                logger.warning("Audio buffer overflow detected!")

            buffer.append(chunk)
            buffer_samples += chunk.shape[0]

            # Speech detected → reset silence counter
            if not detect_silence(chunk, silence_threshold):
                silent_count = 0

                # Enough speech accumulated → yield a full chunk
                if buffer_samples >= min_chunk_samples + overlap_samples:
                    yield_chunk = np.concatenate([overlap_buffer, *buffer], axis=0)
                    logger.debug(f"Yielding async chunk: {len(yield_chunk)/SAMPLE_RATE:.2f}s")
                    yield yield_chunk

                    # Keep overlap for next chunk
                    overlap_buffer = (
                        yield_chunk[-overlap_samples:]
                        if overlap_samples > 0
                        else np.zeros((0, CHANNELS), dtype=DTYPE)
                    )
                    buffer = []
                    buffer_samples = 0
                    await asyncio.sleep(0)

            # Silence detected
            else:
                silent_count += chunk_size

                # Optional auto-close on prolonged silence
                if (
                    auto_close_on_long_silence
                    and silent_count >= silence_frames
                    and buffer_samples >= min_chunk_samples
                ):
                    final_chunk = np.concatenate([overlap_buffer, *buffer], axis=0)
                    if not detect_silence(final_chunk, silence_threshold):
                        yield final_chunk
                    logger.info(
                        f"Auto-closing stream: silence ≥ {silence_duration}s detected"
                    )
                    break

            await asyncio.sleep(0)  # Yield control to event loop

    finally:
        stream.stop()
        stream.close()
        logger.info("Async audio stream closed")


__all__ = [
    "trim_silent_portions",
    "save_chunk",
    "stream_non_silent_audio",
    "async_stream_non_silent_audio",
]