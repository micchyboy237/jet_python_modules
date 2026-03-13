from collections import deque

from fireredvad.core.stream_vad_postprocessor import (
    StreamVadFrameResult,
    StreamVadPostprocessor,
    VadState,
)
from rich.console import Console

console = Console()


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
                    console.print(
                        f"[HYBRID SPLIT] at frame {self.frame_cnt} "
                        f"(soft={self.soft_limit}, hard={self.hard_limit})",
                        style="bold magenta",
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
