import sounddevice as sd
import numpy as np
from typing import Generator, Optional
from tqdm import tqdm
from jet.audio.record_mic import SAMPLE_RATE, CHANNELS, DTYPE, detect_silence, calibrate_silence_threshold
from jet.logger import logger


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
