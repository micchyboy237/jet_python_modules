"""
ShortSegmentAccumulator
-----------------------
Merges adjacent speech segments based on configurable duration and gap thresholds.

LOGIC:
1. Segments ≥ MAX_SEG_DURATION pass through immediately (already complete utterances)
2. Shorter segments are accumulated into groups while:
   - Gap to previous ≤ MAX_GAP
   - Running total ≤ ACC_MAX_DURATION
3. Group is flushed when:
   - Next gap > MAX_GAP, OR
   - Adding next would exceed ACC_MAX_DURATION, OR
   - Group total ≥ MAX_SEG_DURATION (target utterance size reached), OR
   - Trailing true silence exceeds MAX_SILENCE_BEFORE_FLUSH (speaker stopped talking)
"""

from __future__ import annotations

import copy
from typing import List, Tuple

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_ACC_MAX_DURATION_SEC,
    DEFAULT_MAX_SEG_DURATION_SEC,
    DEFAULT_MAX_SEG_GAP_SEC,
    DEFAULT_MAX_SILENCE_BEFORE_FLUSH_SEC,
    DEFAULT_MIN_SEG_DURATION_SEC,
)
from jet.audio.helpers.silence import calibrate_silence_threshold, detect_silence
from rich.console import Console

console = Console()
AccumulatorResult = List[Tuple[SpeechSegment, np.ndarray]]


class ShortSegmentAccumulator:
    """
    Merges adjacent speech segments that are close enough in time and
    short enough to benefit from being combined into longer utterances.

    Parameters
    ----------
    min_seg_duration_sec:
        Minimum duration for a segment to be valid. Segments below this
        are still processed but logged as suspiciously short.
    max_seg_duration_sec:
        Target utterance duration. Segments ≥ this pass through immediately.
        Groups flush once they reach this threshold.
    max_gap_sec:
        Maximum silence gap between segments to allow merging.
    acc_max_duration_sec:
        Hard ceiling on total merged duration. Groups flush before exceeding this.
    max_silence_before_flush_sec:
        If the trailing audio of the pending group contains this much true
        silence (energy below threshold), auto-flush the group. This prevents
        holding segments when the speaker has clearly stopped.
    sample_rate:
        Audio sample rate for concatenation.
    verbose:
        Enable debug logging.
    """

    def __init__(
        self,
        min_seg_duration_sec: float = DEFAULT_MIN_SEG_DURATION_SEC,
        max_seg_duration_sec: float = DEFAULT_MAX_SEG_DURATION_SEC,
        max_gap_sec: float = DEFAULT_MAX_SEG_GAP_SEC,
        acc_max_duration_sec: float = DEFAULT_ACC_MAX_DURATION_SEC,
        max_silence_before_flush_sec: float = DEFAULT_MAX_SILENCE_BEFORE_FLUSH_SEC,
        sample_rate: int = 16_000,
        verbose: bool = False,
    ) -> None:
        self.min_seg_duration = min_seg_duration_sec
        self.max_seg_duration = max_seg_duration_sec
        self.max_gap = max_gap_sec
        self.acc_max_duration = acc_max_duration_sec
        self.max_silence_before_flush = max_silence_before_flush_sec
        self.sample_rate = sample_rate
        self.verbose = verbose

        self._pending: list[tuple[SpeechSegment, np.ndarray]] = []
        self._pending_duration: float = 0.0

        # Cache silence threshold once
        self._silence_threshold = calibrate_silence_threshold()

        if self.verbose:
            console.print(
                f"[dim][Accumulator] init: "
                f"min_seg={min_seg_duration_sec:.2f}s  "
                f"max_seg={max_seg_duration_sec:.2f}s  "
                f"max_gap={max_gap_sec:.2f}s  "
                f"acc_max={acc_max_duration_sec:.2f}s  "
                f"max_silence_flush={max_silence_before_flush_sec:.2f}s  "
                f"sample_rate={sample_rate}  "
                f"silence_threshold={self._silence_threshold:.6f}[/dim]"
            )

    # ─────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────

    def push(
        self,
        segment: SpeechSegment,
        audio_np: np.ndarray,
    ) -> AccumulatorResult:
        """
        Accept one segment. Returns merged groups ready for dispatch.
        """
        seg_dur = float(segment.get("duration", 0.0))
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", 0.0))
        seg_id = segment.get("segment_id", "?")

        if self.verbose:
            console.print(
                f"[bold cyan]─── push: {seg_id} ───[/bold cyan]\n"
                f"  duration={seg_dur:.3f}s  start={seg_start:.3f}s  end={seg_end:.3f}s"
            )

        # ── BUG: negative duration ──────────────────────────────────
        if seg_dur < 0:
            if self.verbose:
                console.print(
                    f"[bold red][Accumulator] BUG: segment {seg_id} has NEGATIVE duration "
                    f"({seg_dur:.3f}s) — start={seg_start:.3f} > end={seg_end:.3f} — SKIPPING[/bold red]"
                )
            return []

        # ── BUG: end before start ───────────────────────────────────
        if seg_end < seg_start:
            if self.verbose:
                console.print(
                    f"[bold red][Accumulator] BUG: segment {seg_id} has end < start "
                    f"(start={seg_start:.3f}s, end={seg_end:.3f}s) — SKIPPING[/bold red]"
                )
            return []

        results: AccumulatorResult = []

        # ── PASS-THROUGH: already a complete utterance ──────────────
        if seg_dur >= self.max_seg_duration:
            if self.verbose:
                console.print(
                    f"  [yellow]≥ max_seg ({self.max_seg_duration:.2f}s) → PASS-THROUGH (standalone)[/yellow]"
                )
            results.extend(self._flush_group())
            results.append((segment, audio_np))
            if self.verbose:
                console.print("[bold cyan]─── /push (pass-through) ───[/bold cyan]")
            return results

        # ── Calculate gap to previous segment ───────────────────────
        gap = None
        last_seg_end = None
        if self._pending:
            last_seg, _ = self._pending[-1]
            last_seg_end = float(last_seg.get("end", 0.0))
            gap = seg_start - last_seg_end
            if self.verbose:
                console.print(
                    f"  pending group: {len(self._pending)} segs, "
                    f"total={self._pending_duration:.3f}s, "
                    f"last_end={last_seg_end:.3f}s, "
                    f"gap={gap:.3f}s"
                )
        else:
            if self.verbose:
                console.print("  pending group: [empty]")

        # ── DECISION: flush current group before adding? ────────────
        should_flush = False
        flush_reason = ""

        if self._pending and gap is not None:
            # Check 1: Gap too large
            if gap > self.max_gap:
                should_flush = True
                flush_reason = f"gap {gap:.3f}s > max_gap {self.max_gap:.2f}s"
            # Check 2: Would exceed hard ceiling
            elif self._pending_duration + seg_dur > self.acc_max_duration:
                would_be = self._pending_duration + seg_dur
                should_flush = True
                flush_reason = (
                    f"would exceed acc_max ({self._pending_duration:.3f}s + "
                    f"{seg_dur:.3f}s = {would_be:.3f}s > {self.acc_max_duration:.3f}s)"
                )
            # Check 3: Current group already reached target size
            elif self._pending_duration >= self.max_seg_duration:
                should_flush = True
                flush_reason = (
                    f"group reached max_seg ({self._pending_duration:.3f}s ≥ "
                    f"{self.max_seg_duration:.3f}s)"
                )
            # Check 4: Gap is negative (overlapping / out-of-order segments)
            elif gap < 0:
                should_flush = True
                flush_reason = (
                    f"NEGATIVE gap ({gap:.3f}s) — segments overlap or out of order "
                    f"(incoming start={seg_start:.3f}s < last_end={last_seg_end:.3f}s)"
                )

        if should_flush:
            if self.verbose:
                console.print(f"  [yellow]Flushing before add: {flush_reason}[/yellow]")
            results.extend(self._flush_group())

        # ── Add segment to pending group ────────────────────────────
        self._pending.append((segment, audio_np))
        self._pending_duration += seg_dur

        segs_in_group = [s.get("segment_id", "?") for s, _ in self._pending]
        if self.verbose:
            console.print(
                f"  [green]ADDED to group {segs_in_group}: "
                f"dur={seg_dur:.3f}s → total={self._pending_duration:.3f}s "
                f"({len(self._pending)} segs)[/green]"
            )

        # ── Check 5: Trailing true silence in the last segment ─────
        silence_flushed = self._check_trailing_silence_and_flush(results)

        # ── Check 6: Group reached target size after adding ────────
        if not silence_flushed and self._pending_duration >= self.max_seg_duration:
            if self.verbose:
                console.print(
                    f"  [yellow]Group reached max_seg after add "
                    f"({self._pending_duration:.3f}s ≥ {self.max_seg_duration:.3f}s) → flush[/yellow]"
                )
            results.extend(self._flush_group())

        if self.verbose:
            console.print(
                f"[bold cyan]─── /push ({len(results)} results) ───[/bold cyan]"
            )
        return results

    def flush(self) -> AccumulatorResult:
        """
        Flush all remaining pending segments as a merged group.
        Called at end of stream.
        """
        if self.verbose:
            console.print(
                f"[bold cyan]─── flush (end of stream) ───[/bold cyan]\n"
                f"  pending: {len(self._pending)} segs, {self._pending_duration:.3f}s"
            )
        return self._flush_group()

    def has_pending(self) -> bool:
        """True if there are buffered segments not yet emitted."""
        return bool(self._pending)

    @property
    def pending_duration(self) -> float:
        """Total duration of buffered (not yet emitted) segments."""
        return self._pending_duration

    # ─────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _check_trailing_silence_and_flush(self, results: AccumulatorResult) -> bool:
        """
        Check the trailing audio of the last segment in the pending group
        for true silence. If silence exceeds max_silence_before_flush_sec,
        flush the entire group.

        Returns True if a flush occurred, False otherwise.
        """
        if not self._pending:
            return False

        # Only check the audio of the LAST segment in the pending group
        last_seg, last_audio = self._pending[-1]

        trailing_silence_sec = self._measure_trailing_silence(last_audio)

        if self.verbose:
            console.print(
                f"  [dim]trailing silence check: "
                f"{trailing_silence_sec:.3f}s "
                f"(threshold={self.max_silence_before_flush:.3f}s)[/dim]"
            )

        if trailing_silence_sec >= self.max_silence_before_flush:
            if self.verbose:
                console.print(
                    f"  [yellow]Trailing silence {trailing_silence_sec:.3f}s ≥ "
                    f"max_silence_flush {self.max_silence_before_flush:.3f}s → flush group[/yellow]"
                )
            results.extend(self._flush_group())
            return True

        return False

    def _measure_trailing_silence(self, audio_np: np.ndarray) -> float:
        """
        Measure the duration of trailing true silence in an audio array.

        Scans backwards from the end of the audio in 10ms frames (160 samples
        at 16kHz) and counts how many consecutive frames are silent.

        Returns duration in seconds.
        """
        if audio_np.size == 0:
            return 0.0

        frame_length = 160  # 10ms at 16kHz
        hop_length = 160  # non-overlapping

        silent_frames = 0
        # Scan backwards from the end
        pos = len(audio_np) - frame_length
        while pos >= 0:
            frame = audio_np[pos : pos + frame_length]
            if detect_silence(frame, self._silence_threshold):
                silent_frames += 1
                pos -= hop_length
            else:
                break

        trailing_silence_sec = (silent_frames * hop_length) / self.sample_rate
        return trailing_silence_sec

    def _flush_group(self) -> AccumulatorResult:
        """Merge current pending group into a single segment and clear buffer."""
        if not self._pending:
            return []

        n = len(self._pending)

        # ── Single segment: no merging needed ─────────────────────────
        if n == 1:
            seg, audio = self._pending[0]
            seg_id = seg.get("segment_id", "?")
            dur = float(seg.get("duration", 0.0))
            start_s = float(seg.get("start", 0.0))
            end_s = float(seg.get("end", 0.0))

            if self.verbose:
                console.print(
                    f"  [dim]── flush single: {seg_id} "
                    f"[{start_s:.3f}→{end_s:.3f}] dur={dur:.3f}s[/dim]"
                )
            self.reset()
            return [(seg, audio)]

        # ── Multiple segments: merge and log details ──────────────────
        seg_ids_before = [s.get("segment_id", "?") for s, _ in self._pending]
        durations_before = [float(s.get("duration", 0.0)) for s, _ in self._pending]
        starts_before = [float(s.get("start", 0.0)) for s, _ in self._pending]
        ends_before = [float(s.get("end", 0.0)) for s, _ in self._pending]
        gaps_between = []
        for i in range(1, len(starts_before)):
            gaps_between.append(starts_before[i] - ends_before[i - 1])

        if self.verbose:
            console.print(
                f"  [bold]── merging {n} segments into group:[/bold]\n"
                f"    IDs:       {seg_ids_before}\n"
                f"    durations: {[f'{d:.3f}s' for d in durations_before]} "
                f"(total={sum(durations_before):.3f}s)\n"
                f"    ranges:    {[f'[{s:.3f}→{e:.3f}]' for s, e in zip(starts_before, ends_before)]}\n"
                f"    gaps:      {[f'{g:.3f}s' for g in gaps_between] if gaps_between else 'N/A'}"
            )

        merged_seg, merged_audio = self._merge_group()
        self.reset()

        if self.verbose:
            console.print(
                f"  [bold green]✅ MERGED → "
                f"[{merged_seg['start']:.3f}→{merged_seg['end']:.3f}] "
                f"dur={merged_seg['duration']:.3f}s "
                f"audio={len(merged_audio)} samples[/bold green]"
            )
        return [(merged_seg, merged_audio)]

    def _merge_group(self) -> tuple[SpeechSegment, np.ndarray]:
        """Combine all pending (segment, audio) pairs into one merged segment."""
        segs = [s for s, _ in self._pending]
        audios = [a for _, a in self._pending]

        first = segs[0]
        last = segs[-1]

        # ── Concatenate audio ───────────────────────────────────────
        merged_audio = np.concatenate(audios, axis=0)

        # ── Build merged segment metadata ───────────────────────────
        merged_seg: SpeechSegment = copy.deepcopy(first)
        merged_seg["start"] = first["start"]
        merged_seg["end"] = last["end"]
        merged_seg["duration"] = float(last["end"]) - float(first["start"])

        # Combine probabilities from all sub-segments
        all_probs: list[float] = []
        for seg in segs:
            all_probs.extend(seg.get("segment_probs", []))
        merged_seg["segment_probs"] = all_probs

        if all_probs:
            merged_seg["prob"] = float(np.mean(all_probs))
            merged_seg["frames_length"] = len(all_probs)
        else:
            merged_seg["prob"] = 0.0
            merged_seg["frames_length"] = 0

        # ── Preserve timestamps ─────────────────────────────────────
        if "start_time_utc" in first:
            merged_seg["start_time_utc"] = first["start_time_utc"]
        if "end_time_utc" in last:
            merged_seg["end_time_utc"] = last["end_time_utc"]

        # ── Track overflow across all sub-segments ──────────────────
        merged_seg["had_overflow"] = any(s.get("had_overflow", False) for s in segs)

        # ── Validate merged segment ─────────────────────────────────
        merged_dur = merged_seg["duration"]
        if merged_dur <= 0:
            if self.verbose:
                console.print(
                    f"  [bold red]⚠ BUG: merged segment has non-positive duration "
                    f"({merged_dur:.3f}s)! first.start={first['start']:.3f}, "
                    f"last.end={last['end']:.3f}[/bold red]"
                )
        if merged_dur > self.acc_max_duration:
            if self.verbose:
                console.print(
                    f"  [yellow]⚠ merged segment ({merged_dur:.3f}s) exceeds "
                    f"acc_max ({self.acc_max_duration:.3f}s)[/yellow]"
                )

        return merged_seg, merged_audio

    def reset(self) -> None:
        """Clear the pending buffer without flushing. Used when the recording session is reset."""
        n = len(self._pending)
        if n > 0:
            console.print(
                f"[yellow][Accumulator] reset: discarding {n} pending segments "
                f"({self._pending_duration:.3f}s)[/yellow]"
            )
        self._pending = []
        self._pending_duration = 0.0
