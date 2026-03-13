import logging
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
from fireredvad.core.constants import (
    FRAME_LENGTH_SAMPLE,
    FRAME_PER_SECONDS,
    FRAME_SHIFT_SAMPLE,
    SAMPLE_RATE,
)
from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stream_vad")

MODEL_DIR = str(
    Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD").expanduser().resolve()
)

audio_queue: queue.Queue[np.ndarray] = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        logger.warning(status)
    audio_queue.put(indata.copy())


def main():
    config = FireRedStreamVadConfig(
        use_gpu=False,
        smooth_window_size=5,
        speech_threshold=0.4,
        pad_start_frame=5,
        min_speech_frame=8,
        max_speech_frame=500,  # ~5 seconds
        min_silence_frame=20,
    )

    stream_vad = FireRedStreamVad.from_pretrained(MODEL_DIR, config)

    logger.info("Starting microphone stream...")

    current_segment_start = None

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        callback=audio_callback,
        blocksize=FRAME_SHIFT_SAMPLE,
    ):
        buffer = np.zeros(0, dtype=np.int16)

        while True:
            chunk = audio_queue.get().flatten()

            buffer = np.concatenate([buffer, chunk])

            while len(buffer) >= FRAME_LENGTH_SAMPLE:
                frame = buffer[:FRAME_LENGTH_SAMPLE]
                buffer = buffer[FRAME_SHIFT_SAMPLE:]

                result = stream_vad.detect_frame(frame)

                if result.is_speech_start:
                    current_segment_start = result.speech_start_frame
                    start_sec = current_segment_start / FRAME_PER_SECONDS

                    logger.info(
                        f"[VAD START] frame={current_segment_start} "
                        f"time={start_sec:.2f}s"
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

                    if frames >= config.max_speech_frame:
                        logger.warning(
                            "[VAD SPLIT] max_speech_frame reached "
                            f"{config.max_speech_frame} frames "
                            f"(~{config.max_speech_frame / FRAME_PER_SECONDS:.2f}s)"
                        )

                    current_segment_start = None


if __name__ == "__main__":
    main()
