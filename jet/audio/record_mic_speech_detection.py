import torch
import wave
import numpy as np
from silero_vad import get_speech_timestamps, load_silero_vad
import sounddevice as sd
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from rich.table import Table

from jet.logger import logger
from jet.audio.helpers.silence import (
    SAMPLE_RATE,
    DTYPE,
    CHANNELS,
    calibrate_silence_threshold,
    detect_silence,
    trim_silent_chunks,
)

# Ensure best results: Silero VAD is reliable only at 16kHz, mono audio
# Change the constants in jet/audio/helpers/silence.py to:
# SAMPLE_RATE = 16000, DTYPE = 'int16', CHANNELS = 1

silero_model = load_silero_vad(onnx=False)


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
    global silero_model
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

                if len(audio_data) % 5 == 0:
                    trimmed_acc = trim_silent_chunks(audio_data, silence_threshold)
                    audio_tensor = convert_audio_to_tensor(trimmed_acc)
                    speech_ts = get_speech_timestamps(
                        audio=audio_tensor,
                        model=silero_model,
                    )
                    display_segments(speech_ts)


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


def convert_audio_to_tensor(audio_data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy audio array or list of chunks to torch tensor suitable for Silero VAD.
    - Ensures mono
    - Converts to float32 in range [-1.0, 1.0]
    - Requires 16kHz input!
    """
    # Accept either a single np.ndarray or a list of chunks
    if isinstance(audio_data, list):
        audio = np.concatenate(audio_data, axis=0)
    else:
        audio = np.asarray(audio_data)

    # Normalize integer PCM to float32 in [-1, 1]
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    elif audio.dtype == np.float64:
        audio = audio.astype(np.float32)
    # If already float, ensure [-1, 1]
    elif np.issubdtype(audio.dtype, np.floating):
        audio = np.clip(audio, -1.0, 1.0)
    else:
        raise ValueError("Unsupported audio dtype")

    tensor = torch.from_numpy(audio)

    # Convert to mono if multi-channel (average channels)
    if tensor.ndim > 1:
        tensor = tensor.mean(dim=1)

    # Sanity checks
    assert tensor.abs().max() <= 1.0 + 1e-5, "Audio not normalized!"
    assert SAMPLE_RATE == 16000, "Wrong sample rate for Silero VAD: must be 16000 Hz"

    return tensor  # shape: (N_samples,), float32, [-1, 1], 16kHz


def display_segments(speech_ts):
    # Convert to seconds and print with rich table
    segments_sec = [
        {"start": t["start"] / SAMPLE_RATE, "end": t["end"] / SAMPLE_RATE}
        for t in speech_ts
    ]

    # Derive total recorded seconds as the end of the last speech segment, or 0 if none
    if segments_sec:
        recorded_seconds = max(seg["end"] for seg in segments_sec)
    else:
        recorded_seconds = 0.0

    table = Table(title=f"Speech segments (total {recorded_seconds:.1f}s recorded)")
    table.add_column("Segment", style="cyan")
    table.add_column("Start (s)", justify="right")
    table.add_column("End (s)", justify="right")
    table.add_column("Duration (s)", justify="right")

    for i, seg in enumerate(segments_sec, 1):
        table.add_row(
            str(i),
            f"{seg['start']:.2f}",
            f"{seg['end']:.2f}",
            f"{seg['end'] - seg['start']:.2f}",
        )

    from rich import print as rprint
    rprint("\n", table, "\n")
