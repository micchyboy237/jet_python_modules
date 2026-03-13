from collections import deque

from fireredvad.core.stream_vad_postprocessor import (
    StreamVadFrameResult,
    StreamVadPostprocessor,
    VadState,
)
from rich.console import Console

console = Console()


class HybridStreamVadPostprocessor(StreamVadPostprocessor):
    """
    Hybrid VAD postprocessor that combines:

    - classic state machine segmentation
    - soft duration splitting using probability valleys
    - hard duration limit for safety

    Designed for real-time streaming ASR pipelines.
    """

    def __init__(
        self,
        smooth_window_size,
        speech_threshold,
        pad_start_frame,
        min_speech_frame,
        max_speech_frame,
        min_silence_frame,
    ):
        # Duration limits
        self.hard_limit = max_speech_frame
        self.soft_limit = int(max_speech_frame * 0.7)

        # Valley detection
        self.search_window = min(100, max_speech_frame // 4)
        self.valley_threshold = 0.6

        # Track probabilities during speech only
        self.speech_probs: deque[float] = deque(maxlen=self.search_window)

        # restart control
        self._force_next_start = False

        # logging trackers
        self._last_state = None
        self._last_silence_cnt = -1

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

        self.speech_probs.clear()
        self._force_next_start = False

        self._last_state = None
        self._last_silence_cnt = -1

    # -------------------------------------------------------------
    # Frame processing
    # -------------------------------------------------------------

    def process_one_frame(self, raw_prob: float) -> StreamVadFrameResult:
        assert 0.0 <= raw_prob <= 1.0

        self.frame_cnt += 1

        smoothed_prob = self.smooth_prob(raw_prob)

        is_speech = bool(self.apply_threshold(smoothed_prob))

        result = StreamVadFrameResult(
            frame_idx=self.frame_cnt,
            is_speech=is_speech,
            raw_prob=round(raw_prob, 3),
            smoothed_prob=round(smoothed_prob, 3),
        )

        return self.state_transition(is_speech, smoothed_prob, result)

    # -------------------------------------------------------------
    # State machine
    # -------------------------------------------------------------

    def state_transition(
        self, is_speech: bool, prob: float, result: StreamVadFrameResult
    ) -> StreamVadFrameResult:
        prev_state = self.state
        prev_speech = self.speech_cnt
        prev_silence = self.silence_cnt

        # ---------------------------------------------------------
        # Restart after forced split
        # ---------------------------------------------------------

        if self._force_next_start:
            result.is_speech_start = True
            result.speech_start_frame = self.frame_cnt

            console.print(
                f"[RECOVERY] {self.frame_cnt:5d} | FORCED SPLIT → NEW SPEECH",
                style="cyan bold",
            )

            self.last_speech_start_frame = result.speech_start_frame
            self._force_next_start = False

        # ---------------------------------------------------------
        # SILENCE
        # ---------------------------------------------------------

        if self.state == VadState.SILENCE:
            if is_speech:
                self.state = VadState.POSSIBLE_SPEECH
                self.speech_cnt = 1
                self.silence_cnt = 0
            else:
                self.silence_cnt += 1

        # ---------------------------------------------------------
        # POSSIBLE SPEECH
        # ---------------------------------------------------------

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

                    console.print(
                        f"[START] {self.frame_cnt:5d} | SPEECH START",
                        style="green bold",
                    )

            else:
                self.state = VadState.SILENCE
                self.speech_cnt = 0
                self.silence_cnt = 1

        # ---------------------------------------------------------
        # SPEECH
        # ---------------------------------------------------------

        elif self.state == VadState.SPEECH:
            if is_speech:
                self.speech_cnt += 1
                self.silence_cnt = 0

                # track speech probabilities only
                self.speech_probs.append(prob)

                force_split = False

                # hard limit
                if self.speech_cnt >= self.hard_limit:
                    force_split = True
                    split_reason = "hard limit"

                # soft limit valley search
                elif (
                    self.speech_cnt >= self.soft_limit
                    and len(self.speech_probs) == self.search_window
                ):
                    valley_prob = min(self.speech_probs)

                    if valley_prob < self.valley_threshold:
                        force_split = True
                        split_reason = f"valley {valley_prob:.3f}"

                if force_split:
                    console.print(
                        f"[SPLIT] {self.frame_cnt:5d} | {split_reason} "
                        f"(speech_cnt={self.speech_cnt})",
                        style="bold red",
                    )

                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame

                    self.last_speech_end_frame = result.speech_end_frame
                    self.last_speech_start_frame = -1

                    # reset counters
                    self.speech_cnt = 0
                    self.speech_probs.clear()

                    # move to silence so next frame starts new segment
                    self.state = VadState.SILENCE
                    self._force_next_start = True

            else:
                self.state = VadState.POSSIBLE_SILENCE
                self.silence_cnt = 1

        # ---------------------------------------------------------
        # POSSIBLE SILENCE
        # ---------------------------------------------------------

        elif self.state == VadState.POSSIBLE_SILENCE:
            if is_speech:
                self.state = VadState.SPEECH
                self.silence_cnt = 0
                self.speech_cnt += 1

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
                    self.speech_probs.clear()

                    console.print(
                        f"[END] {self.frame_cnt:5d} | SPEECH → SILENCE",
                        style="blue bold",
                    )

        # ---------------------------------------------------------
        # Logging
        # ---------------------------------------------------------

        self._log_if_changed(prev_state, prev_speech, prev_silence)

        return result

    # -------------------------------------------------------------
    # Logging helper
    # -------------------------------------------------------------

    def _log_if_changed(self, prev_state, prev_speech, prev_silence):
        if self.state != prev_state:
            style = {
                VadState.SPEECH: "green bold",
                VadState.POSSIBLE_SILENCE: "yellow",
                VadState.POSSIBLE_SPEECH: "cyan",
                VadState.SILENCE: "dim bright_black",
            }.get(self.state, "white")

            console.print(
                f"[STATE] {self.frame_cnt:5d} | {prev_state.name:13} → {self.state.name:13}",
                style=style,
            )

        elif (
            self.state == VadState.SILENCE
            and self.silence_cnt != self._last_silence_cnt
        ):
            if (
                self.silence_cnt % 500 == 0
                or self.silence_cnt == self.min_silence_frame
            ):
                console.print(
                    f"[silence] {self.frame_cnt:5d} continuing ({self.silence_cnt} frames)",
                    style="dim bright_black",
                )

        self._last_state = self.state
        self._last_silence_cnt = self.silence_cnt
