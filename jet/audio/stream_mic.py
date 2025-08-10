import sounddevice as sd
import numpy as np
from typing import Generator, Dict, Tuple, Optional
from tqdm import tqdm
from jet.audio.record_mic import SAMPLE_RATE, CHANNELS, DTYPE, detect_silence, calibrate_silence_threshold, save_wav_file
from jet.logger import logger


def trim_silent_portions(chunk: np.ndarray, silence_threshold: float, sub_chunk_duration: float = 0.1) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Trim silent portions from the start and end of an audio chunk, preserving overlap.
    Args:
        chunk: Audio chunk to trim (numpy array, including overlap).
        silence_threshold: Energy threshold for silence detection.
        sub_chunk_duration: Duration of sub-chunks for silence detection (seconds).
    Returns:
        Tuple of (trimmed chunk or None if all silent, start index, end index).
    """
    sub_chunk_size = int(SAMPLE_RATE * sub_chunk_duration)
    if chunk.shape[0] < sub_chunk_size:
        logger.debug(f"Chunk too small for trimming: {chunk.shape[0]} samples")
        return chunk, 0, chunk.shape[0] if not detect_silence(chunk, silence_threshold) else (None, 0, 0)

    # Split chunk into sub-chunks
    sub_chunks = [chunk[i:i + sub_chunk_size]
                  for i in range(0, chunk.shape[0], sub_chunk_size)]
    start_idx = 0
    end_idx = len(chunk)

    # Find first non-silent sub-chunk
    for i, sub_chunk in enumerate(sub_chunks):
        if len(sub_chunk) >= sub_chunk_size and not detect_silence(sub_chunk, silence_threshold):
            start_idx = i * sub_chunk_size
            break
    else:
        logger.debug("All sub-chunks are silent")
        return None, 0, 0

    # Find last non-silent sub-chunk
    for i in range(len(sub_chunks) - 1, -1, -1):
        if len(sub_chunks[i]) >= sub_chunk_size and not detect_silence(sub_chunks[i], silence_threshold):
            end_idx = (i + 1) * sub_chunk_size
            break

    trimmed_chunk = chunk[start_idx:end_idx]
    logger.debug(
        f"Trimmed {start_idx} samples from start, {chunk.shape[0] - end_idx} from end, remaining: {len(trimmed_chunk)}")
    return trimmed_chunk, start_idx, end_idx


def save_chunk(chunk: np.ndarray, chunk_index: int, timestamp: str, cumulative_duration: float, silence_threshold: float, overlap_samples: int, output_dir: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Save a trimmed audio chunk to a WAV file, including overlap, and return metadata."""
    trimmed_chunk, start_idx, end_idx = trim_silent_portions(
        chunk, silence_threshold)
    if trimmed_chunk is None or len(trimmed_chunk) == 0:
        logger.debug(
            f"Chunk {chunk_index} is entirely silent after trimming, not saved")
        return None, None

    chunk_filename = f"{output_dir}/stream_chunk_{timestamp}_{chunk_index:04d}.wav"
    save_wav_file(chunk_filename, trimmed_chunk)
    chunk_duration = len(trimmed_chunk) / SAMPLE_RATE
    logger.debug(
        f"Saved chunk {chunk_index} to {chunk_filename}, size: {len(trimmed_chunk)} samples, duration: {chunk_duration:.2f}s, "
        f"overlap: {overlap_samples if chunk_index > 0 else 0} samples")
    metadata = {
        "chunk_index": chunk_index,
        "filename": chunk_filename,
        "duration_s": round(chunk_duration, 3),
        "timestamp": timestamp,
        "sample_count": len(trimmed_chunk),
        "start_time_s": round(cumulative_duration, 3),
        "end_time_s": round(cumulative_duration + chunk_duration, 3),
        "trimmed_samples_start": start_idx,
        "trimmed_samples_end": chunk.shape[0] - end_idx,
        "overlap_samples": overlap_samples if chunk_index > 0 else 0
    }
    return chunk_filename, metadata


def stream_non_silent_audio(
    silence_threshold: Optional[float] = None,
    chunk_duration: float = 0.5,
    silence_duration: float = 2.0,
    min_chunk_duration: float = 1.0,
    overlap_duration: float = 0.0
) -> Generator[np.ndarray, None, None]:
    """
    Stream non-silent audio chunks from microphone in real-time using a generator.
    Args:
        silence_threshold: Energy threshold for silence detection. If None, calibrates automatically.
        chunk_duration: Duration of each internal audio chunk in seconds (default: 0.5).
        silence_duration: Duration of silence to stop streaming (default: 2.0).
        min_chunk_duration: Minimum duration of non-overlapping portion of yielded chunks (default: 1.0).
        overlap_duration: Duration of overlap between consecutive chunks in seconds (default: 0.0).
    Yields:
        np.ndarray: Non-silent audio chunk with at least min_chunk_duration + overlap_duration.
    """
    silence_threshold = silence_threshold if silence_threshold is not None else calibrate_silence_threshold()
    chunk_size = int(SAMPLE_RATE * chunk_duration)
    silence_frames = int(silence_duration * SAMPLE_RATE)
    min_chunk_samples = int(SAMPLE_RATE * min_chunk_duration)
    overlap_samples = int(SAMPLE_RATE * overlap_duration)
    silent_count = 0
    chunk_count = 0
    buffer = []
    buffer_samples = 0
    overlap_buffer = np.array([], dtype=DTYPE).reshape(0, CHANNELS)

    logger.info(
        f"Starting real-time audio streaming: {CHANNELS} channel{'s' if CHANNELS > 1 else ''}, "
        f"internal chunk duration {chunk_duration}s, min chunk duration {min_chunk_duration}s, "
        f"overlap duration {overlap_duration}s, silence threshold {silence_threshold:.6f}, "
        f"silence duration {silence_duration}s"
    )

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=chunk_size
    )
    stream.start()

    with tqdm(desc="Streaming chunks", unit="chunk", leave=True) as pbar:
        try:
            while True:
                chunk, overflowed = stream.read(chunk_size)
                if overflowed:
                    logger.warning(
                        f"Buffer overflow detected in chunk {chunk_count}, possible data loss")
                if chunk.shape[0] != chunk_size:
                    logger.warning(
                        f"Chunk {chunk_count} size mismatch: expected {chunk_size}, got {chunk.shape[0]}")

                buffer.append(chunk)
                buffer_samples += chunk.shape[0]

                if not detect_silence(chunk, silence_threshold):
                    silent_count = 0
                    if buffer_samples >= min_chunk_samples:
                        # Prepend overlap from previous chunk
                        yield_chunk = np.concatenate(
                            [overlap_buffer, *buffer], axis=0)
                        chunk_count += 1
                        chunk_duration_yielded = len(yield_chunk) / SAMPLE_RATE
                        logger.debug(
                            f"Yielding non-silent chunk {chunk_count}, size: {len(yield_chunk)} samples, "
                            f"duration: {chunk_duration_yielded:.2f}s, overlap: {len(overlap_buffer)} samples"
                        )
                        yield yield_chunk
                        pbar.update(1)
                        # Update overlap buffer for next chunk
                        overlap_buffer = yield_chunk[-overlap_samples:] if overlap_samples > 0 else np.array(
                            [], dtype=DTYPE).reshape(0, CHANNELS)
                        buffer = []
                        buffer_samples = 0
                else:
                    silent_count += chunk_size
                    logger.debug(
                        f"Silent chunk detected, silent count: {silent_count}/{silence_frames} samples")
                    if silent_count >= silence_frames:
                        # Yield any remaining non-silent buffer
                        if buffer_samples >= min_chunk_samples and not detect_silence(np.concatenate(buffer, axis=0), silence_threshold):
                            yield_chunk = np.concatenate(
                                [overlap_buffer, *buffer], axis=0)
                            chunk_count += 1
                            chunk_duration_yielded = len(
                                yield_chunk) / SAMPLE_RATE
                            logger.debug(
                                f"Yielding final non-silent chunk {chunk_count}, size: {len(yield_chunk)} samples, "
                                f"duration: {chunk_duration_yielded:.2f}s, overlap: {len(overlap_buffer)} samples"
                            )
                            yield yield_chunk
                            pbar.update(1)
                        logger.info(
                            f"Silence detected for {silence_duration}s, stopping stream after {chunk_count} chunks")
                        break
                pbar.set_postfix({
                    "chunks": chunk_count,
                    "silent_samples": silent_count,
                    "buffered_s": buffer_samples / SAMPLE_RATE,
                    "overlap_s": len(overlap_buffer) / SAMPLE_RATE
                })
        finally:
            stream.stop()
            stream.close()
            logger.info(
                f"Audio stream closed, total chunks streamed: {chunk_count}")
