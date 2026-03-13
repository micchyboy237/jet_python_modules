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
from collections import deque
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
    StreamVadPostprocessor,
    VadState,
)
from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig
from jet.audio.audio_waveform.circular_buffer_advanced import CircularBuffer

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


# ==================== HYBRID POSTPROCESSOR (drop-in replacement) ====================
class HybridStreamVadPostprocessor(StreamVadPostprocessor):
    def __init__(
        self,
        smooth_window_size,
        speech_threshold,
        pad_start_frame,
        min_speech_frame,
        max_speech_frame,  # kept for API compatibility
        min_silence_frame,
    ):
        self.soft_limit = 420  # start waiting for natural split
        self.hard_limit = 600  # absolute safety (was 500)
        self.search_window = 100  # ~1 second look-back
        self.valley_threshold = 0.6  # probability dip below this = good split point

        # must exist before parent __init__ calls reset()
        self.recent_probs: deque[float] = deque(maxlen=1024)

        super().__init__(
            smooth_window_size,
            speech_threshold,
            pad_start_frame,
            min_speech_frame,
            max_speech_frame,
            min_silence_frame,
        )

    def reset(self):
        super().reset()
        # ensure deque always exists even during early initialization
        if hasattr(self, "recent_probs"):
            self.recent_probs.clear()
        else:
            self.recent_probs = deque(maxlen=1024)

    def process_one_frame(self, raw_prob: float) -> StreamVadFrameResult:
        assert 0.0 <= raw_prob <= 1.0
        self.frame_cnt += 1

        smoothed_prob = self.smooth_prob(raw_prob)
        self.recent_probs.append(smoothed_prob)  # ← key for valley search

        is_speech = self.apply_threshold(smoothed_prob)

        result = StreamVadFrameResult(
            frame_idx=self.frame_cnt,
            is_speech=is_speech,
            raw_prob=round(raw_prob, 3),
            smoothed_prob=round(smoothed_prob, 3),
        )

        return self.state_transition(is_speech, result)

    def state_transition(
        self, is_speech: bool, result: StreamVadFrameResult
    ) -> StreamVadFrameResult:
        # === UNCHANGED parts (SILENCE, POSSIBLE_SPEECH, POSSIBLE_SILENCE) ===
        if self.hit_max_speech:
            result.is_speech_start = True
            result.speech_start_frame = self.frame_cnt
            self.last_speech_start_frame = result.speech_start_frame
            self.hit_max_speech = False

        if self.state == VadState.SILENCE:
            if is_speech:
                self.state = VadState.POSSIBLE_SPEECH
                self.speech_cnt += 1
            else:
                self.silence_cnt += 1
                self.speech_cnt = 0

        elif self.state == VadState.POSSIBLE_SPEECH:
            if is_speech:
                self.speech_cnt += 1
                if self.speech_cnt >= self.min_speech_frame:
                    self.state = VadState.SPEECH
                    result.is_speech_start = True
                    result.speech_start_frame = max(
                        1,
                        self.frame_cnt - self.speech_cnt + 1 - self.pad_start_frame,
                        self.last_speech_end_frame + 1,
                    )
                    self.last_speech_start_frame = result.speech_start_frame
                    self.silence_cnt = 0
            else:
                self.state = VadState.SILENCE
                self.silence_cnt = 1
                self.speech_cnt = 0

        # === HYBRID SPEECH STATE (this is the ~40-line smart upgrade) ===
        elif self.state == VadState.SPEECH:
            self.speech_cnt += 1
            if is_speech:
                self.silence_cnt = 0

                # Smart split logic (valley + soft + hard)
                force_split = False
                if self.speech_cnt >= self.hard_limit:
                    force_split = True
                elif (
                    self.speech_cnt > self.soft_limit
                    and len(self.recent_probs) >= self.search_window
                ):
                    window = list(self.recent_probs)[-self.search_window :]
                    if min(window) < self.valley_threshold:
                        force_split = True

                if force_split:
                    self.hit_max_speech = True
                    self.speech_cnt = 0
                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame
                    self.last_speech_start_frame = -1
                    self.last_speech_end_frame = result.speech_end_frame
                    logger.debug(
                        f"[HYBRID SPLIT] at frame {self.frame_cnt} "
                        f"(soft={self.soft_limit}, hard={self.hard_limit})"
                    )

            else:
                self.state = VadState.POSSIBLE_SILENCE
                self.silence_cnt += 1

        # POSSIBLE_SILENCE (updated to use hard_limit for consistency)
        elif self.state == VadState.POSSIBLE_SILENCE:
            self.speech_cnt += 1
            if is_speech:
                self.state = VadState.SPEECH
                self.silence_cnt = 0
                if self.speech_cnt >= self.hard_limit:  # ← updated
                    self.hit_max_speech = True
                    self.speech_cnt = 0
                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame
                    self.last_speech_start_frame = -1
                    self.last_speech_end_frame = result.speech_end_frame
            else:
                self.silence_cnt += 1
                if self.silence_cnt >= self.min_silence_frame:
                    self.state = VadState.SILENCE
                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame
                    self.last_speech_end_frame = result.speech_end_frame
                    self.last_speech_start_frame = -1
                    self.speech_cnt = 0

        return result


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
