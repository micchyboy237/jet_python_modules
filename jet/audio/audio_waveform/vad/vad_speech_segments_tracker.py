"""
vad_speech_segments_tracker.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reusable class for accumulating live-streaming VAD frames into SpeechSegment
objects.  One segment is emitted when any of the five end conditions fires:

  1a. "silence"    — prob stayed below threshold for >= min_silence_sec
  1b. "silence"    — past soft_limit AND currently in a silence run
  2a. "valley"     — past soft_limit AND a clean valley trough was found
  2b. "valley"     — past hard_limit (min_valley_duration_s relaxed to 0.5 s)
  3.  "hard_limit" — past hard_limit, safety fallback (no trough needed)

Reuses:
  _compute_preroll          (from vad_speech_segments_extractor)
  _compute_postroll         (from vad_speech_segments_extractor)
  _apply_limit_splits  (from vad_speech_segments_extractor)
  get_best_valley_trough    (from vad_extractors)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
from jet.audio.audio_waveform.vad import vad_logging
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_HARD_LIMIT_MIN_TROUGH_OFFSET_S,
    DEFAULT_HARD_LIMIT_MIN_VALLEY_DURATION_S,
    DEFAULT_HARD_LIMIT_SEC,
    DEFAULT_HARD_LIMIT_SMOOTHING_WINDOW,
    DEFAULT_HARD_LIMIT_TROUGH_PROMINENCE,
    DEFAULT_INCLUDE_NON_SPEECH,
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_MIN_SPEECH_SEC,
    DEFAULT_POSTROLL_HYBRID_THRESHOLD,
    DEFAULT_POSTROLL_MAX_SEC,
    DEFAULT_PREROLL_HYBRID_THRESHOLD,
    DEFAULT_PREROLL_MAX_SEC,
    DEFAULT_PROB_WEIGHT,
    DEFAULT_RETURN_SECONDS,
    DEFAULT_RMS_WEIGHT,
    DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
    DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
    DEFAULT_SOFT_LIMIT_SEC,
    DEFAULT_SOFT_LIMIT_SEC_HIGH,
    DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
    DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
    DEFAULT_THRESHOLD,
    DEFAULT_WITH_SCORES,
)
from jet.audio.audio_waveform.vad.vad_speech_segments_extractor import (
    _apply_limit_splits,
    _compute_postroll,
    _compute_preroll,
)
from jet.audio.helpers.config import HOP_SIZE, HOP_STEP_S, SAMPLE_RATE
from jet.audio.speech.vad_extractors import (
    extract_valley_troughs_from_np_audio,
    get_best_valley_trough,
)
from jet.audio.speech.vad_types import ValleyTrough
from rich.console import Console
from rich.table import Table

console = Console()


EndReason = Literal["silence", "valley", "hard_limit", "none"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal accumulation state
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _TrackerState:
    """All mutable state for one in-progress speech segment."""

    # Raw accumulated data
    probs: List[float] = field(default_factory=list)
    audio_chunks: List[np.ndarray] = field(default_factory=list)

    # Frame counters
    global_frame_offset: int = 0  # global frame index of state[0]
    total_frames_seen: int = 0  # monotonic counter across lifetime

    # Silence tracking
    silence_frame_count: int = 0  # consecutive below-threshold frames
    in_silence: bool = False

    # Segment start (in global frames)
    segment_start_frame: int = 0

    # Wall-clock for progress logging
    started_at: float = field(default_factory=time.monotonic)

    def reset(self, global_frame_offset: int) -> None:
        self.probs.clear()
        self.audio_chunks.clear()
        self.global_frame_offset = global_frame_offset
        self.segment_start_frame = global_frame_offset
        self.silence_frame_count = 0
        self.in_silence = False
        self.started_at = time.monotonic()

    # ── derived helpers ────────────────────────────────────────────────────

    @property
    def n_frames(self) -> int:
        return len(self.probs)

    @property
    def duration_s(self) -> float:
        return self.n_frames * HOP_STEP_S

    @property
    def audio_np(self) -> np.ndarray:
        if not self.audio_chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.audio_chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Public result type
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SegmentResult:
    """Emitted by the tracker when a segment boundary is detected."""

    segments: List[SpeechSegment]
    """One or more SpeechSegment dicts (>1 when soft-limit splits apply)."""

    end_reason: EndReason
    """Why the segment was closed."""

    end_condition_label: str
    """Human-readable condition code, e.g. '1a', '2b', '3'."""

    duration_s: float
    """Total duration of the emitted audio in seconds."""

    probs: List[float]
    """Per-frame probabilities for the segment (same length as audio frames)."""

    audio_np: np.ndarray
    """Concatenated audio for the full segment (before pre/post-roll trim)."""

    valley_trough: Optional[ValleyTrough] = None
    """Valley trough dict from get_best_valley_trough, present when end_reason='valley'."""

    def __repr__(self) -> str:
        return (
            f"<SegmentResult reason={self.end_reason!r} "
            f"label={self.end_condition_label!r} "
            f"duration={self.duration_s:.2f}s "
            f"n_segments={len(self.segments)}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main tracker class
# ─────────────────────────────────────────────────────────────────────────────


class VadSpeechSegmentsTracker:
    """
    Stateful accumulator for live-streamed VAD frames.

    Usage
    -----
    tracker = VadSpeechSegmentsTracker(...)
    for prob, audio_chunk in stream:
        result = tracker.push(prob, audio_chunk)
        if result is not None:
            handle(result)              # segment boundary reached
    final = tracker.flush()             # drain any remaining audio
    if final is not None:
        handle(final)

    Parameters
    ----------
    threshold             Speech/silence decision threshold (default 0.5).
    min_silence_sec       Consecutive silence needed to close on 1a (default 0.8 s).
    min_speech_sec        Minimum segment length to emit (default 0.25 s).
    soft_limit_sec        Soft ceiling — triggers valley / fast-silence logic.
    hard_limit_sec        Hard ceiling — forces close even without a valley.
    sample_rate           Audio sample rate (default 16 000 Hz).
    return_seconds        Segment start/end expressed in seconds (else samples).
    with_scores           Populate segment_probs in emitted SpeechSegment.
    include_non_speech    Emit non-speech gaps between segments.
    preroll_max_sec       Maximum pre-roll window.
    preroll_hybrid_threshold  Hybrid score threshold for pre-roll inclusion.
    preroll_prob_weight   Prob weight in hybrid pre-roll score.
    preroll_rms_weight    RMS weight in hybrid pre-roll score.
    postroll_max_sec      Maximum post-roll window.
    postroll_hybrid_threshold Hybrid score threshold for post-roll inclusion.
    postroll_prob_weight  Prob weight in hybrid post-roll score.
    postroll_rms_weight   RMS weight in hybrid post-roll score.
    soft_limit_min_valley_duration_s   Min valley width for condition 2a.
    soft_limit_smoothing_window        Smoothing window for valley detection.
    soft_limit_trough_prominence       Min trough prominence for valley detection.
    soft_limit_min_trough_offset_s     Min offset from segment start for trough.
    verbose               Enable rich logging (default True).
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        min_silence_sec: float = DEFAULT_MIN_SILENCE_SEC,
        min_speech_sec: float = DEFAULT_MIN_SPEECH_SEC,
        soft_limit_sec: float = DEFAULT_SOFT_LIMIT_SEC,
        soft_limit_sec_high: float = DEFAULT_SOFT_LIMIT_SEC_HIGH,
        hard_limit_sec: float = DEFAULT_HARD_LIMIT_SEC,
        sample_rate: int = SAMPLE_RATE,
        return_seconds: bool = DEFAULT_RETURN_SECONDS,
        with_scores: bool = DEFAULT_WITH_SCORES,
        include_non_speech: bool = DEFAULT_INCLUDE_NON_SPEECH,
        # pre-roll
        preroll_max_sec: float = DEFAULT_PREROLL_MAX_SEC,
        preroll_hybrid_threshold: float = DEFAULT_PREROLL_HYBRID_THRESHOLD,
        preroll_prob_weight: float = DEFAULT_PROB_WEIGHT,
        preroll_rms_weight: float = DEFAULT_RMS_WEIGHT,
        # post-roll
        postroll_max_sec: float = DEFAULT_POSTROLL_MAX_SEC,
        postroll_hybrid_threshold: float = DEFAULT_POSTROLL_HYBRID_THRESHOLD,
        postroll_prob_weight: float = DEFAULT_PROB_WEIGHT,
        postroll_rms_weight: float = DEFAULT_RMS_WEIGHT,
        # valley / soft-limit params
        soft_limit_min_valley_duration_s: float = DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
        soft_limit_smoothing_window: int = DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
        soft_limit_trough_prominence: float = DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
        soft_limit_min_trough_offset_s: float = DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
        # valley / hard-limit params
        hard_limit_min_valley_duration_s: float = DEFAULT_HARD_LIMIT_MIN_VALLEY_DURATION_S,
        hard_limit_smoothing_window: int = DEFAULT_HARD_LIMIT_SMOOTHING_WINDOW,
        hard_limit_trough_prominence: float = DEFAULT_HARD_LIMIT_TROUGH_PROMINENCE,
        hard_limit_min_trough_offset_s: float = DEFAULT_HARD_LIMIT_MIN_TROUGH_OFFSET_S,
        verbose: bool = True,
    ) -> None:
        # Config
        self.threshold = threshold
        self.min_silence_sec = min_silence_sec
        self.min_speech_sec = min_speech_sec
        self.soft_limit_sec = soft_limit_sec
        self.soft_limit_sec_high = soft_limit_sec_high
        self.hard_limit_sec = hard_limit_sec
        self.sample_rate = sample_rate
        self.return_seconds = return_seconds
        self.with_scores = with_scores
        self.include_non_speech = include_non_speech

        self.preroll_max_sec = preroll_max_sec
        self.preroll_hybrid_threshold = preroll_hybrid_threshold
        self.preroll_prob_weight = preroll_prob_weight
        self.preroll_rms_weight = preroll_rms_weight

        self.postroll_max_sec = postroll_max_sec
        self.postroll_hybrid_threshold = postroll_hybrid_threshold
        self.postroll_prob_weight = postroll_prob_weight
        self.postroll_rms_weight = postroll_rms_weight

        self.soft_limit_min_valley_duration_s = soft_limit_min_valley_duration_s
        self.soft_limit_smoothing_window = soft_limit_smoothing_window
        self.soft_limit_trough_prominence = soft_limit_trough_prominence
        self.soft_limit_min_trough_offset_s = soft_limit_min_trough_offset_s

        self.hard_limit_min_valley_duration_s = hard_limit_min_valley_duration_s
        self.hard_limit_smoothing_window = hard_limit_smoothing_window
        self.hard_limit_trough_prominence = hard_limit_trough_prominence
        self.hard_limit_min_trough_offset_s = hard_limit_min_trough_offset_s

        self.verbose = verbose

        # Derived thresholds (frame counts)
        self._min_silence_frames = int(min_silence_sec / HOP_STEP_S)
        self._min_speech_frames = int(min_speech_sec / HOP_STEP_S)

        # Lifetime counters
        self._total_frames_ever: int = 0
        self._segments_emitted: int = 0

        # Accumulation state
        self._state = _TrackerState()
        self._is_accumulating: bool = False  # True once first speech frame seen

        if verbose:
            self._log_config()

    # ── Public API ────────────────────────────────────────────────────────────

    def push(
        self,
        prob: float,
        audio_chunk: Optional[np.ndarray] = None,
    ) -> Optional[SegmentResult]:
        """
        Feed one VAD frame into the tracker.

        Parameters
        ----------
        prob        Smoothed speech probability for this frame [0, 1].
        audio_chunk Raw audio samples for this frame (optional; enables
                    hybrid pre/post-roll and valley scoring when provided).

        Returns
        -------
        SegmentResult if a boundary was detected, else None.
        """
        self._total_frames_ever += 1
        frame_idx = self._total_frames_ever

        is_speech = prob >= self.threshold

        # ── 1. Start new accumulation when speech begins ───────────────────
        if not self._is_accumulating:
            if is_speech:
                self._is_accumulating = True
                self._state.reset(global_frame_offset=frame_idx - 1)
                if self.verbose:
                    console.print(
                        f"[bold cyan]▶ Segment start[/bold cyan] "
                        f"[dim]frame={frame_idx} "
                        f"t={self._state.global_frame_offset * HOP_STEP_S:.2f}s[/dim]"
                    )
            else:
                return None  # still in inter-segment silence

        # ── 2. Accumulate frame ────────────────────────────────────────────
        self._state.probs.append(float(prob))
        if audio_chunk is not None and len(audio_chunk) > 0:
            self._state.audio_chunks.append(audio_chunk.astype(np.float32))

        # ── 3. Update silence run counter ─────────────────────────────────
        if is_speech:
            self._state.silence_frame_count = 0
            self._state.in_silence = False
        else:
            self._state.silence_frame_count += 1
            self._state.in_silence = True

        duration = self._state.duration_s

        # ── 4. Evaluate end conditions (in priority order) ─────────────────
        result = self._evaluate_end_conditions(duration, frame_idx)
        if result is not None:
            return result

        # ── 5. Progress tick ──────────────────────────────────────────────
        if self.verbose and self._state.n_frames % 50 == 0:
            self._log_progress(frame_idx, prob, duration)

        return None

    def flush(self) -> Optional[SegmentResult]:
        """
        Force-emit any remaining accumulated audio as a segment.

        Call this at end-of-stream regardless of whether a boundary was
        reached.  Returns None if fewer than min_speech_sec are accumulated.
        """
        if not self._is_accumulating or self._state.n_frames < self._min_speech_frames:
            if self.verbose and self._is_accumulating:
                console.print(
                    "[yellow]flush: segment too short "
                    f"({self._state.duration_s:.2f}s < {self.min_speech_sec}s) — discarded[/yellow]"
                )
            self._reset_accumulation()
            return None

        if self.verbose:
            console.print(
                f"[bold yellow]⏹ flush[/bold yellow] "
                f"[dim]accumulated {self._state.duration_s:.2f}s — emitting[/dim]"
            )
        return self._emit("hard_limit", "flush")

    def get_current_duration_s(self) -> float:
        """
        Return the duration of the currently accumulating speech segment in seconds.

        Returns 0.0 if no segment is currently being accumulated.
        """
        if not self._is_accumulating:
            return 0.0
        return self._state.duration_s

    def reset(self) -> None:
        """Fully reset all state including lifetime counters."""
        self._total_frames_ever = 0
        self._segments_emitted = 0
        self._reset_accumulation()
        if self.verbose:
            console.print("[bold red]Tracker reset[/bold red]")

    # ── End-condition evaluation ──────────────────────────────────────────────

    def _evaluate_end_conditions(
        self,
        duration: float,
        frame_idx: int,
    ) -> Optional[SegmentResult]:
        """
        Check all five end conditions in priority order.
        Returns a SegmentResult on the first match, else None.

        Thresholds
        ----------
        past_soft      : duration >= soft_limit_sec        (default  6 s)
        past_soft_high : duration >= soft_limit_sec_high   (default 10 s)
        past_hard      : duration >= hard_limit_sec        (default 15 s)  ← safety only
        """
        past_soft = duration >= self.soft_limit_sec
        past_soft_high = duration >= self.soft_limit_sec_high
        past_hard = duration >= self.hard_limit_sec

        silence_long_enough = (
            self._state.silence_frame_count >= self._min_silence_frames
        )

        # 1a: normal silence close (before soft limit)
        if silence_long_enough and not past_soft:
            if self.verbose:
                vad_logging.log_cond_1a(
                    self._state.silence_frame_count,
                    self.soft_limit_sec,
                    frame_idx,
                    duration,
                )
            return self._maybe_emit("silence", "1a", frame_idx, duration)

        # 1b: silence close after soft limit
        if past_soft and self._state.in_silence and silence_long_enough:
            if self.verbose:
                vad_logging.log_cond_1b(
                    self.soft_limit_sec,
                    self._state.in_silence,
                    self._state.silence_frame_count,
                    frame_idx,
                    duration,
                )
            return self._maybe_emit("silence", "1b", frame_idx, duration)

        # 2a: past soft_limit, not yet past soft_limit_high — look for a valley
        if past_soft and not past_soft_high:
            if self.verbose:
                vad_logging.log_cond_2a_check(
                    self.soft_limit_sec, self.soft_limit_sec_high, frame_idx, duration
                )
            valley_troughs = extract_valley_troughs_from_np_audio(
                self._state.audio_np,
                smoothing_window=self.soft_limit_smoothing_window,
                trough_prominence=self.soft_limit_trough_prominence,
                min_valley_duration_s=self.soft_limit_min_valley_duration_s,
                min_trough_offset_s=self.soft_limit_min_trough_offset_s,
            )
            trough = valley_troughs[-1] if valley_troughs else None
            if trough is not None:
                if self.verbose:
                    console.print(
                        f"[yellow][soft] Using valley_troughs[-1]: {trough}[/yellow]"
                    )
                    vad_logging.log_cond_2a_valley_found(trough, frame_idx)
                return self._emit_at_trough(trough, "valley", "2a")
            else:
                if self.verbose:
                    vad_logging.log_cond_2a_no_valley()

        # 2b: past soft_limit_high — look for a valley with relaxed params
        if past_soft_high:
            if self.verbose:
                vad_logging.log_cond_2b_check(
                    self.soft_limit_sec_high, frame_idx, duration
                )
            valley_troughs = extract_valley_troughs_from_np_audio(
                self._state.audio_np,
                smoothing_window=self.hard_limit_smoothing_window,
                trough_prominence=self.hard_limit_trough_prominence,
                min_valley_duration_s=self.hard_limit_min_valley_duration_s,
                min_trough_offset_s=self.hard_limit_min_trough_offset_s,
            )
            trough = valley_troughs[-1] if valley_troughs else None
            if trough is not None:
                if self.verbose:
                    console.print(
                        f"[yellow][hard] Using valley_troughs[-1]: {trough}[/yellow]"
                    )
                    vad_logging.log_cond_2b_valley_found(trough, frame_idx)
                return self._emit_at_trough(trough, "valley", "2b")
            else:
                if self.verbose:
                    vad_logging.log_cond_2b_no_valley_relaxed()

        # 3: absolute safety fallback — hard_limit_sec exceeded, no trough needed
        if past_hard:
            if self.verbose:
                vad_logging.log_cond_3(frame_idx, duration)
            return self._emit("hard_limit", "3")

        if self.verbose:
            vad_logging.log_cond_none(frame_idx, duration)
        return None

    # ── Valley trough search ─────────────────────────────────────────────────

    def _find_valley_trough(
        self,
        probs,
        smoothing_window,
        trough_prominence,
        min_valley_duration_s,
        min_trough_offset_s,
    ):
        """
        Run get_best_valley_trough over the currently accumulated probs.

        Returns the trough dict or None.
        """
        probs = self._state.probs
        if len(probs) < 4:
            return None
        return get_best_valley_trough(
            probs=probs,
            smoothing_window=smoothing_window,
            trough_prominence=trough_prominence,
            min_valley_duration_s=min_valley_duration_s,
            min_trough_offset_s=min_trough_offset_s,
            # frame_offset=self._state.global_frame_offset,
        )

    # ── Emission helpers ──────────────────────────────────────────────────────

    def _maybe_emit(
        self,
        end_reason: EndReason,
        label: str,
        frame_idx: int,
        duration: float,
    ) -> Optional[SegmentResult]:
        """
        Gate emission on min_speech_sec.  Discards very short segments.
        """
        if self._state.n_frames < self._min_speech_frames:
            if self.verbose:
                console.print(
                    f"[dim yellow]  cond {label}: silence met but segment too short "
                    f"({duration:.2f}s) — discarding[/dim yellow]"
                )
            self._reset_accumulation()
            return None
        return self._emit(end_reason, label)

    def _emit(
        self,
        end_reason: EndReason,
        label: str,
        valley_trough: Optional[ValleyTrough] = None,
    ) -> SegmentResult:
        """Build, log, and return a SegmentResult for the full current buffer."""
        probs = list(self._state.probs)
        audio_np = self._state.audio_np
        duration = self._state.duration_s
        frame_start = self._state.global_frame_offset
        frame_end = frame_start + self._state.n_frames - 1

        segments = self._build_segments(
            probs=probs,
            audio_np=audio_np,
            frame_start=frame_start,
            frame_end=frame_end,
        )

        result = SegmentResult(
            segments=segments,
            end_reason=end_reason,
            end_condition_label=label,
            duration_s=duration,
            probs=probs,
            audio_np=audio_np,
            valley_trough=valley_trough,
        )

        self._segments_emitted += 1
        if self.verbose:
            self._log_emission(result)

        self._reset_accumulation()
        return result

    def _emit_at_trough(
        self,
        trough,
        end_reason: EndReason,
        label: str,
    ) -> SegmentResult:
        """
        Truncate accumulation at the trough point, then emit.

        The frames after the trough remain in the buffer for the next segment
        (they become the head of the following accumulation).
        """
        local_trough_frame: int = trough["frame"]
        # Clamp to valid range
        local_trough_frame = min(local_trough_frame, self._state.n_frames - 1)
        local_trough_frame = max(local_trough_frame, 1)

        if self.verbose:
            console.print(
                f"[magenta]  trough at local frame {local_trough_frame} "
                f"(t={trough['time_s']:.2f}s, prob={trough['prob']:.3f})[/magenta]"
            )

        # Split probs and audio at trough
        head_probs = self._state.probs[:local_trough_frame]
        tail_probs = self._state.probs[local_trough_frame:]

        head_audio = self._split_audio_at_frame(local_trough_frame)

        # Temporarily replace state with head-only content
        saved_global_offset = self._state.global_frame_offset
        saved_total = self._total_frames_ever

        self._state.probs = head_probs
        self._state.audio_chunks = [head_audio] if len(head_audio) > 0 else []

        result = self._emit(end_reason, label, valley_trough=trough)

        # Re-seed with tail so the caller's next push() continues naturally
        if tail_probs:
            tail_offset = saved_global_offset + local_trough_frame
            self._is_accumulating = True
            self._state.reset(global_frame_offset=tail_offset)
            self._state.probs = list(tail_probs)
            # audio tail: already consumed above, leave empty
            # (next push() will supply fresh audio chunks)

        return result

    # ── Segment construction ──────────────────────────────────────────────────

    def _build_segments(
        self,
        probs: List[float],
        audio_np: np.ndarray,
        frame_start: int,
        frame_end: int,
    ) -> List[SpeechSegment]:
        """
        Convert raw accumulated data into a list of SpeechSegment dicts.

        Steps
        -----
        1. Compute start / end times (in samples or seconds).
        2. Apply pre-roll / post-roll boundary extension.
        3. Build the initial SpeechSegment.
        4. Run _apply_limit_splits for any sub-splits using soft_limit_sec_high
        as the ceiling (aligns with live condition 2b threshold).
        """
        start_s = frame_start * HOP_STEP_S
        end_s = (frame_end + 1) * HOP_STEP_S
        onset_sample = int(start_s * self.sample_rate)
        end_sample = int(end_s * self.sample_rate)
        total_samples = len(audio_np)

        preroll_samples = 0
        if len(audio_np) > 0 and onset_sample <= total_samples:
            preroll_samples = _compute_preroll(
                onset_sample=onset_sample,
                audio_np=audio_np,
                probs=probs,
                sample_rate=self.sample_rate,
                max_preroll_sec=self.preroll_max_sec,
                hybrid_threshold=self.preroll_hybrid_threshold,
                prob_weight=self.preroll_prob_weight,
                rms_weight=self.preroll_rms_weight,
            )

        postroll_samples = 0
        if len(audio_np) > 0 and end_sample <= total_samples:
            postroll_samples = _compute_postroll(
                end_sample=end_sample,
                audio_np=audio_np,
                probs=probs,
                sample_rate=self.sample_rate,
                max_postroll_sec=self.postroll_max_sec,
                hybrid_threshold=self.postroll_hybrid_threshold,
                prob_weight=self.postroll_prob_weight,
                rms_weight=self.postroll_rms_weight,
            )

        new_start_sample = max(0, onset_sample - preroll_samples)
        new_end_sample = min(total_samples, end_sample + postroll_samples)
        new_start_s = new_start_sample / self.sample_rate
        new_end_s = new_end_sample / self.sample_rate
        duration_s = new_end_s - new_start_s

        avg_prob = float(np.mean(probs)) if probs else 0.0
        start_val = new_start_s if self.return_seconds else new_start_sample
        end_val = new_end_s if self.return_seconds else new_end_sample

        seg = SpeechSegment(
            num=self._segments_emitted + 1,
            start=start_val,
            end=end_val,
            prob=avg_prob,
            duration=duration_s,
            frames_length=len(probs),
            frame_start=frame_start,
            frame_end=frame_end,
            type="speech",
            segment_probs=list(probs) if self.with_scores else [],
        )

        segments = _apply_limit_splits(
            segments=[seg],
            probs=probs,
            audio_np=audio_np,
            sample_rate=self.sample_rate,
            hop_sec=HOP_STEP_S,
            max_limit_sec=self.soft_limit_sec_high,  # ← was soft_limit_sec
            min_valley_duration_s=self.soft_limit_min_valley_duration_s,
            smoothing_window=self.soft_limit_smoothing_window,
            trough_prominence=self.soft_limit_trough_prominence,
            min_trough_offset_s=self.soft_limit_min_trough_offset_s,
            return_seconds=self.return_seconds,
            with_scores=self.with_scores,
        )
        return segments

    # ── Audio splitting helper ────────────────────────────────────────────────

    def _split_audio_at_frame(self, local_frame: int) -> np.ndarray:
        """
        Return audio samples up to *local_frame* and trim the state's
        audio_chunks to remove those samples.
        """
        audio_np = self._state.audio_np
        split_sample = local_frame * HOP_SIZE
        head = audio_np[:split_sample]
        tail = audio_np[split_sample:]
        self._state.audio_chunks = [tail] if len(tail) > 0 else []
        return head

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_accumulation(self) -> None:
        """Clear accumulated audio / probs; advance the global frame offset."""
        next_offset = self._state.global_frame_offset + self._state.n_frames
        self._state.reset(global_frame_offset=next_offset)
        self._is_accumulating = False

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_config(self) -> None:
        table = Table(title="VadSpeechSegmentsTracker config", show_header=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")
        rows = [
            ("threshold", f"{self.threshold}"),
            ("min_silence_sec", f"{self.min_silence_sec}"),
            ("min_speech_sec", f"{self.min_speech_sec}"),
            ("soft_limit_sec", f"{self.soft_limit_sec}"),
            ("soft_limit_sec_high", f"{self.soft_limit_sec_high}"),
            ("hard_limit_sec", f"{self.hard_limit_sec}"),
            ("sample_rate", f"{self.sample_rate}"),
            ("preroll_max_sec", f"{self.preroll_max_sec}"),
            ("postroll_max_sec", f"{self.postroll_max_sec}"),
            (
                "soft_limit_min_valley_duration_s",
                f"{self.soft_limit_min_valley_duration_s}",
            ),
            ("soft_limit_trough_prominence", f"{self.soft_limit_trough_prominence}"),
        ]
        for k, v in rows:
            table.add_row(k, v)
        console.print(table)

    def _log_progress(self, frame_idx: int, prob: float, duration: float) -> None:
        elapsed = time.monotonic() - self._state.started_at
        soft_pct = min(100.0, duration / self.soft_limit_sec * 100)
        hard_pct = min(100.0, duration / self.hard_limit_sec * 100)
        silence_pct = min(
            100.0,
            self._state.silence_frame_count / self._min_silence_frames * 100,
        )
        console.print(
            f"[dim]  frame={frame_idx:6d} "
            f"t={duration:5.2f}s  "
            f"prob={prob:.3f}  "
            f"sil={silence_pct:5.1f}%  "
            f"soft={soft_pct:5.1f}%  "
            f"hard={hard_pct:5.1f}%  "
            f"wall={elapsed:.1f}s[/dim]"
        )

    def _log_emission(self, result: SegmentResult) -> None:
        label_colour = {
            "1a": "green",
            "1b": "green",
            "2a": "yellow",
            "2b": "yellow",
            "3": "red",
            "flush": "magenta",
        }.get(result.end_condition_label, "white")

        console.print(
            f"[bold {label_colour}]■ Segment #{self._segments_emitted}[/bold {label_colour}] "
            f"emitted  "
            f"[dim]reason=[/dim][bold {label_colour}]{result.end_reason}[/bold {label_colour}] "
            f"[dim]cond=[/dim][bold {label_colour}]{result.end_condition_label}[/bold {label_colour}]  "
            f"duration=[bold]{result.duration_s:.2f}s[/bold]  "
            f"sub_segs=[bold]{len(result.segments)}[/bold]"
        )
