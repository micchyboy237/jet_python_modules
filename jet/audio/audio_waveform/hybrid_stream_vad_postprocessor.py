from collections import deque
from typing import List, Optional

from fireredvad.core.stream_vad_postprocessor import (
    StreamVadFrameResult,
    StreamVadPostprocessor,
    VadState,
)
from jet.audio.speech.vad_peak_analyzer import (
    VADSegment,
    ValleyTrough,
    extract_valley_troughs,
)
from rich.console import Console

console = Console()


class HybridStreamVadPostprocessor(StreamVadPostprocessor):
    def __init__(
        self,
        smooth_window_size: int = 5,
        speech_threshold: float = 0.5,
        pad_start_frame: int = 5,
        min_speech_frame: int = 30,
        soft_max_speech_frame: int = 450,  # start waiting for natural split
        hard_max_speech_frame: int = 800,
        min_silence_frame: int = 20,
        search_window: int = 200,
        valley_threshold: float = 0.65,
        min_valley_consecutive_frames: int = 5,
    ):
        self.soft_limit = soft_max_speech_frame
        self.hard_limit = hard_max_speech_frame
        self.search_window = search_window
        self.valley_threshold = valley_threshold
        self.min_valley_consecutive_frames = min_valley_consecutive_frames

        # Logging suppression helpers
        self._last_state = None
        self._last_speech_cnt = -1
        self._last_silence_cnt = -1
        self._last_force_split_frame = -1

        # must exist before parent __init__ calls reset()
        self.recent_probs: deque[float] = deque(maxlen=1024)

        self.last_force_split_reason = None

        super().__init__(
            smooth_window_size,
            speech_threshold,
            pad_start_frame,
            min_speech_frame,
            hard_max_speech_frame,
            min_silence_frame,
        )

    @property
    def was_force_splitted(self) -> bool:
        return self.last_force_split_reason in ("valley_detection", "hard_limit")

    @property
    def last_split_reason(self) -> Optional[str]:
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

    # ------------------------------------------------------------------
    # Valley detection helpers (replace _has_valid_valley)
    # ------------------------------------------------------------------
    def _build_vad_valleys(
        self, window: List[float], frame_offset: int
    ) -> List[VADSegment]:
        """
        Scan `window` for contiguous runs of frames below valley_threshold
        and return them in (start_frame, end_frame, min_prob, min_idx) format.
        """
        valleys = []
        run_start = None
        run_min_prob = float("inf")
        min_local_idx = -1

        for idx, p in enumerate(window):
            if p < self.valley_threshold:
                if run_start is None:
                    run_start = idx
                    run_min_prob = p
                    min_local_idx = idx
                else:
                    if p < run_min_prob:
                        run_min_prob = p
                        min_local_idx = idx
            else:
                if run_start is not None:
                    run_end = idx - 1
                    frame_length = run_end - run_start + 1
                    if frame_length >= self.min_valley_consecutive_frames:
                        min_abs_frame = frame_offset + min_local_idx
                        start_s = (frame_offset + run_start) * getattr(
                            self, "FRAME_DURATION_S", 0.02
                        )
                        end_s = (frame_offset + run_end) * getattr(
                            self, "FRAME_DURATION_S", 0.02
                        )
                        duration_s = (run_end - run_start + 1) * getattr(
                            self, "FRAME_DURATION_S", 0.02
                        )
                        details = {
                            "troughs": [min_local_idx],
                            "min_prob_frame": min_abs_frame,
                            "min_prob_s": min_abs_frame
                            * getattr(self, "FRAME_DURATION_S", 0.02),
                            "min_probability": run_min_prob,
                        }
                        valley = {
                            "frame_start": frame_offset + run_start,
                            "frame_end": frame_offset + run_end,
                            "frame_length": frame_length,
                            "start_s": start_s,
                            "end_s": end_s,
                            "duration_s": duration_s,
                            "details": details,
                        }
                        valleys.append(valley)
                    run_start = None
                    run_min_prob = float("inf")
                    min_local_idx = -1

        if run_start is not None:
            run_end = len(window) - 1
            frame_length = run_end - run_start + 1
            if frame_length >= self.min_valley_consecutive_frames:
                min_abs_frame = frame_offset + min_local_idx
                start_s = (frame_offset + run_start) * getattr(
                    self, "FRAME_DURATION_S", 0.02
                )
                end_s = (frame_offset + run_end) * getattr(
                    self, "FRAME_DURATION_S", 0.02
                )
                duration_s = (run_end - run_start + 1) * getattr(
                    self, "FRAME_DURATION_S", 0.02
                )
                details = {
                    "troughs": [min_local_idx],
                    "min_prob_frame": min_abs_frame,
                    "min_prob_s": min_abs_frame
                    * getattr(self, "FRAME_DURATION_S", 0.02),
                    "min_probability": run_min_prob,
                }
                valley = {
                    "frame_start": frame_offset + run_start,
                    "frame_end": frame_offset + run_end,
                    "frame_length": frame_length,
                    "start_s": start_s,
                    "end_s": end_s,
                    "duration_s": duration_s,
                    "details": details,
                }
                valleys.append(valley)

        return valleys

    def _select_best_trough(
        self, troughs: List[ValleyTrough]
    ) -> Optional[ValleyTrough]:
        """
        Pick the most appropriate trough to use as the split point.

        Strategy
        --------
        1. Take the **last** (most recent) trough — minimises how much
           speech audio is discarded.
        2. If several troughs share the same valley end frame (unlikely but
           possible), prefer the one with the **lowest probability** (deepest
           silence = cleanest cut).
        3. Returns None if the list is empty.
        """
        if not troughs:
            return None
        # Sort: primary key = valley end frame descending (most recent first),
        #       secondary key = prob ascending (deepest first).
        return sorted(troughs, key=lambda t: (-t["valley"]["frame_end"], t["prob"]))[0]

    def state_transition(
        self, is_speech: bool, result: StreamVadFrameResult
    ) -> StreamVadFrameResult:
        prev_state = self.state
        prev_speech = self.speech_cnt
        prev_silence = self.silence_cnt

        if self.hit_max_speech and is_speech:
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
                window: List[float] = []
                deepest_valley_frame: Optional[int] = None

                dynamic_window = self.speech_cnt // 2

                if self.speech_cnt >= self.hard_limit:
                    force_split = True
                    if len(self.recent_probs) >= self.search_window:
                        window = list(self.recent_probs)[-dynamic_window:]
                        frame_offset = self.frame_cnt - len(window) + 1
                        valleys = self._build_vad_valleys(window, frame_offset)
                        troughs = extract_valley_troughs(valleys, duration_s=0.15)
                        best = self._select_best_trough(troughs)
                        if best is not None:
                            deepest_valley_frame = best["frame"]
                elif (
                    self.speech_cnt > self.soft_limit
                    and len(self.recent_probs) >= self.search_window
                ):
                    window = list(self.recent_probs)[-dynamic_window:]
                    frame_offset = self.frame_cnt - len(window) + 1
                    valleys = self._build_vad_valleys(window, frame_offset)
                    troughs = extract_valley_troughs(valleys, duration_s=0.3)
                    best = self._select_best_trough(troughs)
                    if best is not None:
                        deepest_valley_frame = best["frame"]
                        force_split = True

                if force_split:
                    if window and deepest_valley_frame is not None:
                        best_prob = best["prob"] if "prob" in best else min(window)
                        min_prob_str = (
                            (
                                f"min_prob={best_prob:.3f} "
                                f"(trough frame={deepest_valley_frame}, "
                                f"valley {best['valley']['start_s']:.2f}s"
                                f"–{best['valley']['end_s']:.2f}s)"
                            )
                            if best is not None
                            else f"min_prob={min(window):.3f}"
                        )
                    else:
                        min_prob_str = "hard limit"

                    console.print(
                        f"[SPLIT] {self.frame_cnt:5d} | SPEECH → END ({min_prob_str}, cnt={self.speech_cnt})",
                        style="bold red",
                    )
                    # Record the real split frame (could be a few frames earlier)
                    self._last_force_split_frame = (
                        deepest_valley_frame
                        if deepest_valley_frame is not None
                        else self.frame_cnt
                    )
                    console.print(
                        f" soft={self.soft_limit} hard={self.hard_limit}",
                        style="dim magenta",
                    )
                    self.hit_max_speech = True
                    self.speech_cnt = 0
                    result.is_speech_end = True
                    self.last_force_split_reason = (
                        "valley_detection"
                        if deepest_valley_frame is not None
                        else "hard_limit"
                    )
                    result.speech_end_frame = (
                        deepest_valley_frame
                        if deepest_valley_frame is not None
                        else self.frame_cnt
                    )
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

                # See original comment about avoiding short segments.
                if self.silence_cnt >= self.min_silence_frame:
                    self.state = VadState.SILENCE
                    result.is_speech_end = True
                    self.last_force_split_reason = "silence"
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
            # Log entry into silence specifically
            if self.state == VadState.SILENCE:
                console.print(
                    f"[silence] {self.frame_cnt:5d} entered silence (after {prev_silence} transition frames)",
                    style="dim bright_black",
                )

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

        # Always log starts/ends/splits (already handled above, but can reinforce)
        if result.is_speech_start or result.is_speech_end:
            pass  # already logged in the transition blocks

        # Update trackers
        self._last_state = self.state
        self._last_speech_cnt = self.speech_cnt
        self._last_silence_cnt = self.silence_cnt
