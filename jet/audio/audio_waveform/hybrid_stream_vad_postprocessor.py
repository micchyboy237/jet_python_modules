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
        max_speech_frame,
        min_silence_frame,
    ):
        self.soft_limit = 450  # start waiting for natural split
        self.hard_limit = 800  # absolute safety (was 500)
        self.search_window = 200  # ~2 second look-back
        self.valley_threshold = 0.65  # probability dip below this = good split point

        # Logging suppression helpers
        self._last_state = None
        self._last_speech_cnt = -1
        self._last_silence_cnt = -1
        self._last_force_split_frame = -1

        # must exist before parent __init__ calls reset()
        self.recent_probs: deque[float] = deque(maxlen=1024)

        self.last_force_split_reason = "none"

        super().__init__(
            smooth_window_size,
            speech_threshold,
            pad_start_frame,
            min_speech_frame,
            max_speech_frame,
            min_silence_frame,
        )

    @property
    def was_last_end_forced(self) -> bool:
        return self.hit_max_speech  # only meaningful right after end

    @property
    def last_split_reason(self) -> str:
        return self.last_force_split_reason

    def reset(self):
        super().reset()
        if hasattr(self, "recent_probs"):
            self.recent_probs.clear()
        else:
            self.recent_probs = deque(maxlen=1024)
        # Reset logging trackers on reset
        self._last_state = None
        self._last_speech_cnt = -1
        self._last_silence_cnt = -1
        self._last_force_split_frame = -1

    def process_one_frame(self, raw_prob: float) -> StreamVadFrameResult:
        assert 0.0 <= raw_prob <= 1.0
        self.frame_cnt += 1

        smoothed_prob = self.smooth_prob(raw_prob)
        self.recent_probs.append(smoothed_prob)

        is_speech = bool(self.apply_threshold(smoothed_prob))

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
        prev_state = self.state
        prev_speech = self.speech_cnt
        prev_silence = self.silence_cnt

        if self.hit_max_speech:
            result.is_speech_start = True
            console.print(
                f"[RECOVERY] {self.frame_cnt:5d} | {self.state.name} → SPEECH START (hit max recovery)",
                style="cyan bold",
            )
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
                    console.print(
                        f"[START] {self.frame_cnt:5d} | POSSIBLE_SPEECH → SPEECH (cnt={self.speech_cnt})",
                        style="green bold",
                    )
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

        elif self.state == VadState.SPEECH:
            self.speech_cnt += 1
            if is_speech:
                self.silence_cnt = 0
                force_split = False
                window = []
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
                    min_prob_str = (
                        f"min_prob={min(window):.3f}" if window else "hard limit"
                    )
                    console.print(
                        f"[SPLIT] {self.frame_cnt:5d} | SPEECH → END  ({min_prob_str}, cnt={self.speech_cnt})",
                        style="bold red",
                    )
                    console.print(
                        f"  soft={self.soft_limit}  hard={self.hard_limit}",
                        style="dim magenta",
                    )
                    self.hit_max_speech = True
                    self.last_force_split_reason = (
                        "valley_detection" if window else "hard_limit"
                    )
                    self.speech_cnt = 0
                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame
                    self.last_speech_start_frame = -1
                    self.last_speech_end_frame = result.speech_end_frame
            else:
                self.state = VadState.POSSIBLE_SILENCE
                self.silence_cnt += 1

        elif self.state == VadState.POSSIBLE_SILENCE:
            self.speech_cnt += 1
            if is_speech:
                self.state = VadState.SPEECH
                self.silence_cnt = 0
                if self.speech_cnt >= self.hard_limit:
                    self.hit_max_speech = True
                    self.speech_cnt = 0
                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame
                    self.last_speech_start_frame = -1
                    self.last_speech_end_frame = result.speech_end_frame
                    console.print(
                        f"[STATE] {self.frame_cnt:5d} | "
                        f"POSSIBLE_SILENCE → SPEECH (continued) → hit hard limit → END",
                        style="bold magenta",
                    )
            else:
                self.silence_cnt += 1
                if self.silence_cnt >= self.min_silence_frame:
                    self.state = VadState.SILENCE
                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame
                    self.last_speech_end_frame = result.speech_end_frame
                    console.print(
                        f"[END] {self.frame_cnt:5d} | POSSIBLE_SILENCE → SILENCE (after {self.silence_cnt} silent, speech was {self.frame_cnt - self.last_speech_start_frame} frames)",
                        style="blue bold",
                    )
                    self.last_speech_start_frame = -1
                    self.speech_cnt = 0

        # ── Selective logging ───────────────────────────────────────
        self._log_if_changed(prev_state, prev_speech, prev_silence, result)

        return result

    def _log_if_changed(self, prev_state, prev_speech, prev_silence, result):
        """Only print when state or important counters change"""
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
            # Log silence progression only occasionally
            if (
                self.silence_cnt % 500 == 0
                or self.silence_cnt == self.min_silence_frame
            ):
                console.print(
                    f"[silence] {self.frame_cnt:5d} continuing ({self.silence_cnt} frames)",
                    style="dim bright_black",
                )

        # Always log starts/ends/splits (already handled above, but can reinforce)
        if result.is_speech_start or result.is_speech_end:
            pass  # already logged in the transition blocks

        # Update trackers
        self._last_state = self.state
        self._last_speech_cnt = self.speech_cnt
        self._last_silence_cnt = self.silence_cnt
