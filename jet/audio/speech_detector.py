from typing import Generator, Optional, Tuple

import numpy as np
from fireredvad.core.constants import (
    FRAME_PER_SECONDS,
    FRAME_SHIFT_SAMPLE,
    SAMPLE_RATE,
)
from jet.audio.audio_stream.firered_stream import FireredStream
from jet.audio.audio_stream.vad_worker import (
    VadSegmentWorker,  # re-exported for backward compatibility
)
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_MAX_SPEECH_SEC,
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_MIN_SPEECH_SEC,
    DEFAULT_THRESHOLD,
)
from jet.audio.helpers.circular_audio_buffer import CircularAudioBuffer
from jet.audio.helpers.silence import (
    CHANNELS,
    DTYPE,
    calibrate_silence_threshold,
    detect_silence,
)
from jet.logger import logger
from tqdm import tqdm

DEFAULT_BUFFER_MAX_SEC = 60.0
_HOPS_PER_READ = 50
DEFAULT_TRIM_OVERLAP_SEC = 0.3


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
    verbose: bool = False,
) -> Generator[Tuple[SpeechSegment, np.ndarray], None, None]:
    """
    Record audio from microphone with silence detection and progress tracking.

    ARCHITECTURE (decoupled VAD)
    -----------------------------
    This loop's only job is to drain `FireredStream` as fast as PortAudio
    delivers chunks — append to the buffer, detect gaps/silence, and hand
    off to `VadSegmentWorker` running on its own thread. The expensive VAD
    pass (`extract_current_speech_segment`, which re-scans the whole
    buffer) never runs on this thread, so it can never cause this loop to
    fall behind and starve `FireredStream`'s internal queue — which is
    what was silently dropping audio blocks and showing up as
    "Audio gap detected" warnings.

    Speech segments include absolute UTC timestamps:
        - ``segment["start_time_utc"]``: datetime when speech started
        - ``segment["end_time_utc"]``: datetime when speech ended
    """
    last_gap_check = 0
    silence_threshold = (
        silence_threshold
        if silence_threshold is not None
        else calibrate_silence_threshold()
    )
    duration_str = f"{duration}s" if duration is not None else "indefinite"
    logger.info(
        f"Starting recording: {CHANNELS} channel{'s' if CHANNELS > 1 else ''}\n"
        f"Max duration {duration_str}\n"
        f"VAD runs on a decoupled worker thread — drain loop never blocks on it."
    )
    if not quit_on_silence:
        logger.info(
            f"Silence threshold {silence_threshold:.6f}\nSilence duration {silence_duration}s"
        )

    block_size = FRAME_SHIFT_SAMPLE
    chunk_size = block_size * _HOPS_PER_READ
    max_frames = int(duration * SAMPLE_RATE) if duration is not None else float("inf")
    silence_frames = int(silence_duration * FRAME_PER_SECONDS) * block_size
    grace_frames = FRAME_PER_SECONDS * block_size

    audio_data = CircularAudioBuffer(
        max_sec=buffer_max_sec,
        sample_rate=SAMPLE_RATE,
        dtype=DTYPE,
    )
    worker = VadSegmentWorker(
        audio_data,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
        overlap_seconds=overlap_seconds,
        trim_silent=trim_silent,
        silence_threshold=silence_threshold,
        trim_overlap_sec=DEFAULT_TRIM_OVERLAP_SEC,
        verbose=verbose,
    )
    worker.start()

    silent_count = 0
    true_silence = False
    recorded_frames = 0

    pbar_kwargs = (
        {"total": duration, "desc": "Recording", "unit": "s", "leave": True}
        if duration is not None
        else {"desc": "Recording", "unit": "s", "leave": True}
    )
    with tqdm(**pbar_kwargs) as pbar, FireredStream(latency="high") as stream:
        for chunk, capture_time, overflow in stream:
            if recorded_frames >= max_frames:
                break

            if overflow:
                logger.warning(
                    f"Audio overflow detected! Lost ~{stream.total_samples_lost} samples. "
                    f"Quality may be degraded for subsequent segments."
                )
                worker.notify_overflow()

            audio_data.append(chunk, capture_time=capture_time)
            recorded_frames += chunk_size
            pbar.update(chunk_size / SAMPLE_RATE)

            if audio_data.gap_events and len(audio_data.gap_events) > last_gap_check:
                new_gaps = audio_data.gap_events[last_gap_check:]
                last_gap_check = len(audio_data.gap_events)
                for gap in new_gaps:
                    logger.warning(
                        f"Audio gap: {gap['gap_sec']:.3f}s at "
                        f"expected={gap['expected_time']}, actual={gap['actual_time']}"
                    )
                if audio_data.total_gap_sec > 1.0:
                    logger.error(
                        f"Total gap time exceeded 1 second: {audio_data.total_gap_sec:.3f}s. "
                        f"Audio quality may be degraded."
                    )

            if worker.error is not None:
                raise RuntimeError("VadSegmentWorker failed") from worker.error

            if recorded_frames > grace_frames and detect_silence(
                chunk, silence_threshold
            ):
                silent_count += chunk_size
                if silent_count >= silence_frames:
                    if not true_silence:
                        logger.info(
                            f"Silence detected for {silence_duration}s, stopping recording"
                        )
                        true_silence = True
                    if quit_on_silence:
                        break
                    worker.notify_silence()
                    for segment, seg_audio_np in worker.poll_results():
                        yield segment, seg_audio_np
                    continue
            else:
                silent_count = 0
                true_silence = False

            worker.notify_new_audio()
            for segment, seg_audio_np in worker.poll_results():
                yield segment, seg_audio_np

    # Stream closed — stop the worker thread BEFORE flushing the trailing
    # segment, so flush_final_segment() never races with the worker thread
    # over shared state (prev_segment, curr_speech_segs, etc.).
    for segment, seg_audio_np in worker.stop_and_drain():
        yield segment, seg_audio_np

    worker.flush_final_segment()
    for segment, seg_audio_np in worker.poll_results():
        yield segment, seg_audio_np

    if worker.error is not None:
        raise RuntimeError("VadSegmentWorker failed") from worker.error

    if not audio_data:
        logger.warning("No audio recorded")
        return

    actual_duration = len(audio_data) / SAMPLE_RATE
    logger.info(
        f"Recording complete, actual duration: {actual_duration:.2f}s. "
        f"VAD passes: {worker._vad_passes}, segments emitted: {worker._segments_emitted}"
    )
