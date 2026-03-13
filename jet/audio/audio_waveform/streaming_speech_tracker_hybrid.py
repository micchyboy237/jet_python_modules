# 🔥 Hybrid Strategy Applied to Your FireRedVAD Stream
# Drop-in replacement for StreamVadPostprocessor (~60 lines total new class)
# Uses:
#   soft_limit = 420   → start looking for split opportunities
#   hard_limit = 600   → force split (safety)
#   search_window = 100 → look back ~1s for probability valley
#   valley_threshold = 0.6
#
# Result: natural boundaries instead of hard cuts at exactly 500 frames.
# No retroactive gaps, fully streaming-compatible, keeps all original state machine.

import logging
import os
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
from fireredvad.core.audio_feat import AudioFeat
from fireredvad.core.constants import (
    FRAME_LENGTH_SAMPLE,
    FRAME_PER_SECONDS,
    FRAME_SHIFT_SAMPLE,
    SAMPLE_RATE,
)
from fireredvad.core.detect_model import DetectModel
from fireredvad.core.stream_vad_postprocessor import (
    StreamVadFrameResult,
)
from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig
from jet.audio.audio_waveform.circular_buffer_advanced import CircularBuffer
from jet.audio.audio_waveform.hybrid_stream_vad_postprocessor import (
    HybridStreamVadPostprocessor,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stream_vad")

MODEL_DIR = str(
    Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD").expanduser().resolve()
)

audio_queue: queue.Queue[np.ndarray] = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        logger.warning(status)
    # Already float32 when dtype='float32' is used in InputStream
    audio_queue.put(
        indata[:, 0].copy()
    )  # make 1D, consistent with app_with_speech_tracking


# ==================== MAIN (manual build with hybrid postprocessor) ====================
def main():
    config = FireRedStreamVadConfig(
        use_gpu=False,
        smooth_window_size=5,
        speech_threshold=0.4,
        pad_start_frame=5,
        min_speech_frame=8,
        max_speech_frame=500,  # ← kept for API compatibility, but hybrid uses soft/hard limits
        min_silence_frame=20,
    )

    logger.info("Starting microphone stream with HYBRID VAD...")

    # Manual construction (exactly like from_pretrained but with our hybrid postprocessor)
    cmvn_path = os.path.join(MODEL_DIR, "cmvn.ark")
    feat_extractor = AudioFeat(cmvn_path)

    vad_model = DetectModel.from_pretrained(MODEL_DIR)
    if config.use_gpu:
        vad_model.cuda()
    else:
        vad_model.cpu()

    postprocessor = HybridStreamVadPostprocessor(
        config.smooth_window_size,
        config.speech_threshold,
        config.pad_start_frame,
        config.min_speech_frame,
        config.max_speech_frame,
        config.min_silence_frame,
    )

    stream_vad = FireRedStreamVad(
        audio_feat=feat_extractor,
        vad_model=vad_model,
        postprocessor=postprocessor,
        config=config,
    )

    # ─── Circular buffer for incoming audio samples ───
    MAX_BUFFER_SECONDS = 6.0  # enough headroom for jitter/lag
    MAX_BUFFER_SAMPLES = int(MAX_BUFFER_SECONDS * SAMPLE_RATE)
    audio_buffer = CircularBuffer(capacity=MAX_BUFFER_SAMPLES)

    current_segment_start = None

    BLOCK_SIZE = 512  # Matches typical real-time block size

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=BLOCK_SIZE,
    ):
        while True:
            try:
                chunk = audio_queue.get(timeout=0.1)  # already 1D float32
            except queue.Empty:
                continue

            # Add new audio data to the circular buffer
            audio_buffer.extend(chunk)

            # Process as many full analysis frames as possible
            while audio_buffer.available() >= FRAME_LENGTH_SAMPLE:
                # Get the oldest available frame (contiguous copy)
                frame = audio_buffer.get_frame(FRAME_LENGTH_SAMPLE)
                if frame is None:
                    break  # should not happen due to while condition

                # FireRedVAD expects int16 in [-32768, 32767]
                frame_int16 = np.round(frame * 32767).astype(np.int16)

                result: StreamVadFrameResult = stream_vad.detect_frame(frame_int16)

                if result.is_speech_start:
                    current_segment_start = result.speech_start_frame
                    start_sec = current_segment_start / FRAME_PER_SECONDS
                    logger.info(
                        f"[VAD START] frame={current_segment_start} time={start_sec:.2f}s"
                    )

                if result.is_speech_end:
                    start = result.speech_start_frame
                    end = result.speech_end_frame
                    frames = end - start + 1
                    duration = frames / FRAME_PER_SECONDS
                    start_sec = start / FRAME_PER_SECONDS
                    end_sec = end / FRAME_PER_SECONDS

                    logger.info(
                        f"[VAD END] {start_sec:.2f}s → {end_sec:.2f}s "
                        f"(duration={duration:.2f}s frames={frames})"
                    )

                    if frames >= postprocessor.hard_limit:
                        logger.warning(
                            f"[HYBRID SPLIT] hard_limit reached ({postprocessor.hard_limit} frames)"
                        )
                    elif frames > postprocessor.soft_limit:
                        logger.info("[HYBRID SPLIT] natural valley / silence split")

                    current_segment_start = None

                # Consume one analysis hop (usually 10 ms / 160 samples at 16 kHz)
                audio_buffer.advance(FRAME_SHIFT_SAMPLE)

            # Optional: log buffer fill level every ~10 seconds (for debugging)
            # if audio_buffer.available() > MAX_BUFFER_SAMPLES * 0.9:
            #     logger.warning(f"Buffer almost full: {audio_buffer.available()}/{MAX_BUFFER_SAMPLES}")


if __name__ == "__main__":
    main()
