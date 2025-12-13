import numpy as np
from silero_vad import load_silero_vad
import sounddevice as sd
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

from jet.audio.speech.silero.speech_timestamps_extractor import extract_speech_timestamps
from jet.audio.speech.utils import convert_audio_to_tensor, display_segments
from jet.audio.speech.utils import save_completed_segment
from jet.logger import logger
from jet.audio.helpers.silence import (
    SAMPLE_RATE,
    DTYPE,
    CHANNELS,
    calibrate_silence_threshold,
    detect_silence,
)

silero_model = load_silero_vad(onnx=False)

if not hasattr(__import__(__name__), "record_from_mic_segment_state"):
    record_from_mic_segment_state = {
        "finalized_up_to_sample": 0,
        "last_seen_segments": [],
        "counter": 0,
    }
else:
    record_from_mic_segment_state = getattr(__import__(__name__), "record_from_mic_segment_state")


def record_from_mic(
    duration: Optional[int] = None,
    silence_threshold: Optional[float] = None,
    silence_duration: float = 2.0,
    trim_silent: bool = True,
    *,
    output_dir: Optional[Path | str] = None,
) -> Optional[np.ndarray]:
    """Record audio from microphone with silence detection and progress tracking.

    When ``output_dir`` is supplied each detected speech segments (from Silero VAD)
    is saved under ``<output_dir>/segments/segment_<num>/`` as:
        - sound.wav
        - metadata.json

    Args:
        duration: Maximum recording duration in seconds (None = indefinite).
        silence_threshold: Silence level in RMS (None = auto-calibrated).
        silence_duration: Seconds of continuous silence to stop recording.
        trim_silent: If True, removes silent sections from the final audio.
        output_dir: Optional output directory to save segments and metadata as they are detected.
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

    audio_data: List[np.ndarray] = []
    silent_count = 0
    recorded_frames = 0

    # Segment saving setup
    save_segments = output_dir is not None
    if save_segments:
        output_dir = Path(output_dir)
        segment_root = output_dir / "segments"
        segment_root.mkdir(parents=True, exist_ok=True)
    # Use module/global record_from_mic_segment_state as state
    state = record_from_mic_segment_state
    state["counter"] = 0
    state["finalized_up_to_sample"] = 0
    state["last_seen_segments"] = []
    curr_segment = None
    prev_segment = None

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
                pbar.update(0.5)

                # Skip silence detection during grace period
                if recorded_frames > grace_frames and detect_silence(chunk, silence_threshold):
                    silent_count += chunk_size
                    if silent_count >= silence_frames:
                        logger.info(
                            f"Silence detected for {silence_duration}s, stopping recording")
                        break
                else:
                    silent_count = 0

                # Run Silero VAD every few chunks to detect speech segments in real-time
                # if len(audio_data) % 5 == 0:
                audio_tensor = convert_audio_to_tensor(audio_data)
                speech_ts = extract_speech_timestamps(audio=audio_tensor, model=silero_model)
                display_segments(speech_ts)

                curr_segment = speech_ts[-1] if speech_ts else prev_segment
                if save_segments and curr_segment and prev_segment and curr_segment["start"] != prev_segment["start"]:
                    full_audio_np = np.concatenate(audio_data, axis=0)

                    state["counter"] = save_completed_segment(
                        segment_root=segment_root,
                        counter=state["counter"],
                        ts=prev_segment,
                        audio_np=full_audio_np,
                    )
                prev_segment = curr_segment

    if not audio_data:
        logger.warning("No audio recorded")
        return None

    audio_data = np.concatenate(audio_data, axis=0)

    # === FINAL FLUSH: save any remaining speech ===
    if save_segments:
        full_audio_np = np.concatenate(audio_data, axis=0)  # always untrimmed full recording
        state["counter"] = save_completed_segment(
            segment_root=segment_root,
            counter=state["counter"],
            ts=prev_segment,
            audio_np=full_audio_np,
            trim_silence=trim_silent,           # pass the original flag
            silence_threshold=silence_threshold,  # already calibrated earlier
        )

    actual_duration = len(audio_data) / SAMPLE_RATE
    logger.info(f"Recording complete, actual duration: {actual_duration:.2f}s")
    return audio_data

