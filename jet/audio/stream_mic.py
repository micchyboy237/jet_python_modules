import sounddevice as sd
import numpy as np
from typing import Generator, Optional
from tqdm import tqdm
from jet.audio.record_mic import SAMPLE_RATE, CHANNELS, DTYPE, detect_silence, calibrate_silence_threshold
from jet.logger import logger


def stream_non_silent_audio(
    silence_threshold: Optional[float] = None,
    chunk_duration: float = 0.5,
    silence_duration: float = 2.0
) -> Generator[np.ndarray, None, None]:
    """
    Stream non-silent audio chunks from microphone in real-time using a generator.
    Args:
        silence_threshold: Energy threshold for silence detection. If None, calibrates automatically.
        chunk_duration: Duration of each audio chunk in seconds (default: 0.5).
        silence_duration: Duration of silence to stop streaming (default: 2.0).
    Yields:
        np.ndarray: Non-silent audio chunk.
    """
    silence_threshold = silence_threshold if silence_threshold is not None else calibrate_silence_threshold()
    chunk_size = int(SAMPLE_RATE * chunk_duration)
    silence_frames = int(silence_duration * SAMPLE_RATE)
    silent_count = 0
    chunk_count = 0

    logger.info(
        f"Starting real-time audio streaming: {CHANNELS} channel{'s' if CHANNELS > 1 else ''}, "
        f"chunk duration {chunk_duration}s, silence threshold {silence_threshold:.6f}, "
        f"silence duration {silence_duration}s"
    )

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE
    )
    stream.start()

    with tqdm(desc="Streaming chunks", unit="chunk", leave=True) as pbar:
        try:
            while True:
                chunk = stream.read(chunk_size)[0]
                if not detect_silence(chunk, silence_threshold):
                    silent_count = 0
                    chunk_count += 1
                    logger.debug(
                        f"Yielding non-silent chunk {chunk_count}, size: {len(chunk)} samples")
                    yield chunk
                    pbar.update(1)
                else:
                    silent_count += chunk_size
                    logger.debug(
                        f"Silent chunk detected, silent count: {silent_count}/{silence_frames} samples")
                    if silent_count >= silence_frames:
                        logger.info(
                            f"Silence detected for {silence_duration}s, stopping stream after {chunk_count} chunks")
                        break
                pbar.set_postfix(
                    {"chunks": chunk_count, "silent_samples": silent_count})
        finally:
            stream.stop()
            stream.close()
            logger.info(
                f"Audio stream closed, total chunks streamed: {chunk_count}")
