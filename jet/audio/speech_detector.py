from typing import Generator, List, Optional, Tuple

import numpy as np
import sounddevice as sd
from fireredvad.core.constants import (
    FRAME_PER_SECONDS,
    FRAME_SHIFT_SAMPLE,
    SAMPLE_RATE,
)
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.audio_waveform.vad.vad_speech_segments_extractor import (
    extract_speech_timestamps,
)
from jet.audio.helpers.circular_audio_buffer import CircularAudioBuffer
from jet.audio.helpers.silence import (
    CHANNELS,
    DTYPE,
    calibrate_silence_threshold,
    detect_silence,
)
from jet.audio.speech.utils import display_segments
from jet.logger import logger
from tqdm import tqdm

DEFAULT_THRESHOLD = 0.3
DEFAULT_MIN_SILENCE_SEC = 0.8
DEFAULT_MIN_SPEECH_SEC = 0.250
DEFAULT_MAX_SPEECH_SEC = 12.0
DEFAULT_BUFFER_MAX_SEC = 120.0

# Each stream.read() delivers this many firered hops (10 ms each).
# 50 hops × 10 ms = 0.5 s per read — tune freely, must stay a whole number.
_HOPS_PER_READ = 50


def record_from_mic(
    duration: Optional[int] = None,
    silence_threshold: Optional[float] = None,
    silence_duration: float = 2.0,
    trim_silent: bool = False,
    overlap_seconds: float = 0.0,
    quit_on_silence: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_speech_duration_sec: float = DEFAULT_MAX_SPEECH_SEC,
    buffer_max_sec: float = DEFAULT_BUFFER_MAX_SEC,
) -> Generator[Tuple[SpeechSegment, np.ndarray], None, None]:
    """Record audio from microphone with silence detection and progress tracking.

    Args:
        duration: Maximum recording duration in seconds (None = indefinite).
        silence_threshold: Silence level in RMS (None = auto-calibrated).
        silence_duration: Seconds of continuous silence to stop recording.
        trim_silent: If True, removes silent edges from the segments and final audio.
        overlap_seconds: Seconds of overlap to add from the previous segment when
            yielding a new one (prevents information loss at boundaries).
        buffer_max_sec: Maximum seconds of audio kept in the sliding window.
            Older audio is evicted automatically. Defaults to 120 s.
    """
    silence_threshold = (
        silence_threshold
        if silence_threshold is not None
        else calibrate_silence_threshold()
    )

    duration_str = f"{duration}s" if duration is not None else "indefinite"
    logger.info(
        f"Starting recording: {CHANNELS} channel{'s' if CHANNELS > 1 else ''}\n"
        f"Max duration {duration_str}"
    )
    if not quit_on_silence:
        logger.info(
            f"Silence threshold {silence_threshold:.6f}\nSilence duration {silence_duration}s"
        )

    # block_size aligns the hardware buffer to one firered hop (10 ms = 160 samples).
    # chunk_size is how many samples we pull per loop iteration — always a clean
    # multiple of block_size so the VAD's internal framing stays aligned.
    block_size = FRAME_SHIFT_SAMPLE  # 160 samples = 10 ms
    chunk_size = block_size * _HOPS_PER_READ  # 8000 samples = 0.5 s

    max_frames = int(duration * SAMPLE_RATE) if duration is not None else float("inf")
    silence_frames = int(silence_duration * FRAME_PER_SECONDS) * block_size
    grace_frames = FRAME_PER_SECONDS * block_size  # 1 s of hops before silence kicks in

    audio_data = CircularAudioBuffer(
        max_sec=buffer_max_sec,
        sample_rate=SAMPLE_RATE,
        dtype=DTYPE,
    )

    silent_count = 0
    recorded_frames = 0
    curr_segment: Optional[SpeechSegment] = None
    prev_segment: Optional[SpeechSegment] = None
    last_yielded_end_sec: float = 0.0
    curr_speech_segs: List[SpeechSegment] = []

    pbar_kwargs = (
        {"total": duration, "desc": "Recording", "unit": "s", "leave": True}
        if duration is not None
        else {"desc": "Recording", "unit": "s", "leave": True}
    )

    with tqdm(**pbar_kwargs) as pbar:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=block_size,  # align hardware buffer to one firered hop
        )
        with stream:
            while recorded_frames < max_frames:
                chunk = stream.read(chunk_size)[0]
                audio_data.append(chunk)
                recorded_frames += chunk_size
                pbar.update(chunk_size / SAMPLE_RATE)

                if recorded_frames > grace_frames and detect_silence(
                    chunk, silence_threshold
                ):
                    silent_count += chunk_size
                    if silent_count >= silence_frames:
                        logger.info(
                            f"Silence detected for {silence_duration}s, stopping recording"
                        )
                        if quit_on_silence:
                            break
                        if prev_segment:
                            seg_audio_np = extract_segment_data(
                                prev_segment,
                                audio_data,
                                trim_silent=trim_silent,
                                silence_threshold=silence_threshold,
                            )
                            yield prev_segment, seg_audio_np
                            audio_data.trim_to_sec(float(prev_segment["end"]))
                            prev_segment = None
                        if curr_speech_segs:
                            display_segments(curr_speech_segs, done=True)
                        continue
                else:
                    silent_count = 0

                curr_speech_segs = extract_current_speech_segment(
                    audio_data,
                    threshold=threshold,
                    min_silence_duration_sec=min_silence_duration_sec,
                    min_speech_duration_sec=min_speech_duration_sec,
                    max_speech_duration_sec=max_speech_duration_sec,
                )
                curr_segment = (
                    curr_speech_segs[-1] if curr_speech_segs else prev_segment
                )
                prev_segment = (
                    curr_speech_segs[-2] if len(curr_speech_segs) > 1 else None
                )

                if (
                    curr_segment
                    and prev_segment
                    and curr_segment["start"] != prev_segment["start"]
                ):
                    last_yielded_window_sec = max(
                        0.0, last_yielded_end_sec - audio_data.trimmed_sec
                    )
                    effective_start_sec = max(
                        last_yielded_window_sec,
                        float(prev_segment["start"]) - overlap_seconds,
                    )
                    if effective_start_sec >= float(prev_segment["end"]):
                        prev_segment = curr_segment
                        continue

                    original_start = prev_segment["start"]
                    prev_segment["start"] = effective_start_sec
                    seg_audio_np = extract_segment_data(
                        prev_segment,
                        audio_data,
                        trim_silent=trim_silent,
                        silence_threshold=silence_threshold,
                    )
                    prev_segment["start"] = original_start
                    prev_segment["duration"] = (
                        prev_segment["end"] - prev_segment["start"]
                    )
                    last_yielded_end_sec = (
                        float(prev_segment["end"]) + audio_data.trimmed_sec
                    )
                    yield prev_segment, seg_audio_np
                    audio_data.trim_to_sec(float(prev_segment["end"]))

                prev_segment = curr_segment

    if not audio_data:
        logger.warning("No audio recorded")
        return None

    if prev_segment:
        display_segments(curr_speech_segs)
        last_yielded_window_sec = max(
            0.0, last_yielded_end_sec - audio_data.trimmed_sec
        )
        effective_start_sec = max(
            last_yielded_window_sec,
            float(prev_segment["start"]) - overlap_seconds,
        )
        if effective_start_sec >= float(prev_segment["end"]):
            logger.info(
                "Skipping final segment yield due to overlap causing empty audio"
            )
            return

        prev_segment["start"] = effective_start_sec
        seg_audio_np = extract_segment_data(
            prev_segment,
            audio_data,
            trim_silent=trim_silent,
            silence_threshold=silence_threshold,
        )
        yield prev_segment, seg_audio_np

    actual_duration = len(audio_data) / SAMPLE_RATE
    logger.info(f"Recording complete, actual duration: {actual_duration:.2f}s")


def extract_segment_data(
    segment: SpeechSegment,
    audio_np: CircularAudioBuffer,
    trim_silent: bool = True,
    silence_threshold: Optional[float] = None,
) -> np.ndarray:
    """Extract the audio for *segment* from the circular buffer.

    VAD timestamps are relative to the start of ``audio_np``'s current window,
    so we can pass them directly to ``slice_seconds`` without any index math.
    """
    return audio_np.slice_seconds(
        start_sec=float(segment["start"]),
        end_sec=float(segment["end"]),
        trim_silent=trim_silent,
        silence_threshold=silence_threshold,
    )


def extract_current_speech_segment(
    audio_data: CircularAudioBuffer,
    threshold: float = DEFAULT_THRESHOLD,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_speech_duration_sec: float = DEFAULT_MAX_SPEECH_SEC,
) -> List[SpeechSegment]:
    full_audio_np = np.concatenate(audio_data, axis=0)
    curr_speech_segs, speech_probs = extract_speech_timestamps(
        audio=full_audio_np,
        with_scores=True,
        return_seconds=True,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
    )
    display_segments(curr_speech_segs)
    return curr_speech_segs
