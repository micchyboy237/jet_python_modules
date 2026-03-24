from typing import Optional

import numpy as np
import sounddevice as sd
from jet.audio.helpers.config import SAMPLE_RATE
from jet.audio.helpers.energy import get_audio_duration, has_sound, trim_silent
from jet.audio.utils.loader import load_audio
from jet.logger import logger
from tqdm import tqdm


def record_from_mic(
    duration: Optional[int] = None,
    silence_duration: float = 2.0,
    stop_on_silence: bool = False,
) -> Optional[np.ndarray]:
    """Record audio from microphone with optional silence detection.

    Args:
        duration: Maximum recording duration in seconds (None = indefinite).
        silence_duration: Seconds of continuous silence to stop (only used if stop_on_silence=True).
        stop_on_silence: If True, stop recording after `silence_duration` of silence.
                         If False (default), record indefinitely (or until `duration` or Ctrl+C).
    """

    duration_str = f"{duration}s" if duration is not None else "indefinite"
    stop_mode = f"silence ({silence_duration}s)" if stop_on_silence else "Ctrl+C only"
    logger.info(
        f"Starting recording...\nmax duration: {duration_str} stop mode: {stop_mode}"
    )

    chunk_size = int(SAMPLE_RATE * 0.5)  # 0.5 second chunks
    max_frames = int(duration * SAMPLE_RATE) if duration is not None else float("inf")
    silence_frames = int(silence_duration * SAMPLE_RATE)
    grace_frames = int(SAMPLE_RATE * 1.0)  # 1-second grace period

    audio_data = []
    silent_count = 0
    recorded_frames = 0

    pbar_kwargs = (
        {"total": duration, "desc": "Recording", "unit": "s", "leave": True}
        if duration is not None
        else {"desc": "Recording", "unit": "s", "leave": True}
    )

    with tqdm(**pbar_kwargs) as pbar:
        stream = sd.InputStream(samplerate=SAMPLE_RATE)

        with stream:
            while recorded_frames < max_frames:
                try:
                    chunk, _ = stream.read(chunk_size)  # chunk is (frames, channels)
                    audio_data.append(chunk)
                    recorded_frames += len(chunk)  # use actual frames returned
                    pbar.update(0.5)

                    # === Silence detection (only when enabled) ===
                    if stop_on_silence and recorded_frames > grace_frames:
                        if not has_sound(chunk):
                            silent_count += len(chunk)
                            if silent_count >= silence_frames:
                                logger.info(
                                    f"Silence detected for {silence_duration}s, stopping recording"
                                )
                                break
                        else:
                            silent_count = 0

                except KeyboardInterrupt:
                    logger.info(
                        "Keyboard interrupt (Ctrl+C) received, stopping recording"
                    )
                    break

    if not audio_data:
        logger.warning("No audio recorded")
        return None

    # Concatenate
    full_audio = np.concatenate(audio_data, axis=0)

    audio_mono, _ = load_audio(full_audio, SAMPLE_RATE)

    trimmed_mono = trim_silent(
        samples=audio_mono,
        sample_rate=SAMPLE_RATE,
        frame_length_ms=25.0,
        hop_length_ms=10.0,
    )

    actual_duration = get_audio_duration(trimmed_mono)
    logger.info(f"Recording complete, actual duration: {actual_duration:.2f}s")
    return trimmed_mono if trimmed_mono.size else None
