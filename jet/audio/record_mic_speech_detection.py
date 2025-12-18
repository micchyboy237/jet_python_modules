import numpy as np
import sounddevice as sd
from typing import Generator, List, Optional, Tuple
from tqdm import tqdm
from silero_vad import load_silero_vad

from jet.audio.speech.silero.speech_timestamps_extractor import SpeechSegment, extract_speech_timestamps
from jet.audio.speech.utils import convert_audio_to_tensor, display_segments
from jet.logger import logger
from jet.audio.helpers.silence import (
    SAMPLE_RATE,
    DTYPE,
    CHANNELS,
    calibrate_silence_threshold,
    detect_silence,
    trim_silent_chunks,
)

silero_model = load_silero_vad(onnx=False)

def record_from_mic(
    duration: Optional[int] = None,
    silence_threshold: Optional[float] = None,
    silence_duration: float = 2.0,
    trim_silent: bool = False,
    overlap_seconds: float = 0.5,
    quit_on_silence: bool = False,
) -> Generator[Tuple[SpeechSegment, np.ndarray, np.ndarray], None, None]:
    """Record audio from microphone with silence detection and progress tracking.

    Args:
        duration: Maximum recording duration in seconds (None = indefinite).
        silence_threshold: Silence level in RMS (None = auto-calibrated).
        silence_duration: Seconds of continuous silence to stop recording.
        trim_silent: If True, removes silent edges from the segments and final audio.
        overlap_seconds: Seconds of overlap to add from the previous segment when yielding a new one (prevents information loss at boundaries).
    """
    silence_threshold = silence_threshold if silence_threshold is not None else calibrate_silence_threshold()

    duration_str = f"{duration}s" if duration is not None else "indefinite"
    logger.info(
        f"Starting recording: {CHANNELS} channel{'s' if CHANNELS > 1 else ''}\n"
        f"Max duration {duration_str}"
    )
    if not quit_on_silence:
        logger.info(f"Silence threshold {silence_threshold:.6f}\nSilence duration {silence_duration}s")

    chunk_size = int(SAMPLE_RATE * 0.5)  # 0.5 second chunks
    max_frames = int(duration * SAMPLE_RATE) if duration is not None else float('inf')
    silence_frames = int(silence_duration * SAMPLE_RATE)
    grace_frames = int(SAMPLE_RATE * 1.0)  # 1-second grace period

    audio_data = []
    silent_count = 0
    recorded_frames = 0

    curr_segment: Optional[SpeechSegment] = None
    prev_segment: Optional[SpeechSegment] = None
    last_yielded_end_sample: int = 0
    speech_ts: List[SpeechSegment] = []

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

                        if quit_on_silence:
                            break

                        if prev_segment:
                            seg_audio_np = extract_segment_data(prev_segment, audio_data, trim_silent=trim_silent, silence_threshold=silence_threshold)
                            full_audio_np = np.concatenate(audio_data, axis=0)

                            yield prev_segment, seg_audio_np, full_audio_np

                            # Reset segments
                            prev_segment = None

                        if speech_ts:
                            display_segments(speech_ts)

                        continue
                else:
                    silent_count = 0

                # Run Silero VAD every chunk to detect speech segments in real-time
                speech_ts = extract_and_display_speech_segments(audio_data)
                curr_segment = speech_ts[-1] if speech_ts else prev_segment
                prev_segment = speech_ts[-2] if len(speech_ts) > 1 else None

                if curr_segment and prev_segment and curr_segment["start"] != prev_segment["start"]:
                    # Apply overlap: start the yielded segment earlier by overlap_seconds (but not before last_yielded_end_sample)
                    overlap_samples = int(overlap_seconds * SAMPLE_RATE)
                    effective_start = max(last_yielded_end_sample, int(prev_segment["start"]) - overlap_samples)
                    # Safety: if effective_start >= end, skip yielding this segment to avoid empty audio
                    if effective_start >= int(prev_segment["end"]):
                        prev_segment = curr_segment
                        continue
                    # Temporarily adjust the segment start for audio extraction
                    original_start = prev_segment["start"]
                    prev_segment["start"] = effective_start  # type: ignore

                    seg_audio_np = extract_segment_data(prev_segment, audio_data, trim_silent=trim_silent, silence_threshold=silence_threshold)

                    # Restore original timestamps for the yielded SpeechSegment
                    prev_segment["start"] = original_start  # type: ignore
                    # Update duration/prob if needed (Silero already provides them based on original boundaries)
                    prev_segment["duration"] = (prev_segment["end"] - prev_segment["start"]) / SAMPLE_RATE  # type: ignore
                    # Update the tracking of the last yielded end
                    last_yielded_end_sample = int(prev_segment["end"])

                    full_audio_np = np.concatenate(audio_data, axis=0)

                    yield prev_segment, seg_audio_np, full_audio_np
                prev_segment = curr_segment

    if not audio_data:
        logger.warning("No audio recorded")
        return None

    # === FINAL FLUSH: save any remaining speech ===
    if prev_segment:
        display_segments(speech_ts)
        # Final segment overlap handling
        overlap_samples = int(overlap_seconds * SAMPLE_RATE)
        effective_start = max(last_yielded_end_sample, int(prev_segment["start"]) - overlap_samples)
        if effective_start >= int(prev_segment["end"]):
            logger.info("Skipping final segment yield due to overlap causing empty audio")
            return

        seg_audio_np = extract_segment_data(prev_segment, audio_data, trim_silent=trim_silent, silence_threshold=silence_threshold)
        full_audio_np = np.concatenate(audio_data, axis=0)

        yield prev_segment, seg_audio_np, full_audio_np

    actual_duration = len(audio_data) / SAMPLE_RATE
    logger.info(f"Recording complete, actual duration: {actual_duration:.2f}s")

def extract_segment_data(segment: SpeechSegment, audio_np: List[np.ndarray], trim_silent: bool = True, silence_threshold: Optional[float] = None) -> np.ndarray:
    start_sample = int(segment["start"])
    end_sample = int(segment["end"])

    full_audio_np = np.concatenate(audio_np, axis=0)

    # Extract raw segment (including possible silence at edges)
    seg_audio = full_audio_np[start_sample:end_sample]

    if trim_silent:
        if silence_threshold is None:
            silence_threshold = calibrate_silence_threshold()

        chunk_size = int(0.1 * SAMPLE_RATE)  # 100 ms chunks
        chunks = [seg_audio[i:i + chunk_size] for i in range(0, len(seg_audio), chunk_size)]
        trimmed_chunks = trim_silent_chunks(chunks, silence_threshold)

        if not trimmed_chunks:
            seg_audio = np.array([], dtype=DTYPE)
        else:
            seg_audio = np.concatenate(trimmed_chunks, axis=0)

    return seg_audio

def extract_and_display_speech_segments(audio_data: List[np.ndarray]) -> List[SpeechSegment]:
    audio_tensor = convert_audio_to_tensor(audio_data)
    speech_ts = extract_speech_timestamps(audio=audio_tensor, model=silero_model)
    display_segments(speech_ts)
    return speech_ts
