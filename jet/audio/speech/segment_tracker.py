from typing import List, Optional

from jet.audio.speech.silero.speech_types import SpeechSegment


class SpeechSegmentTracker:
    """Tracks speech segments from Silero VAD and determines when a completed segment should be yielded.

    Handles overlap between consecutive segments and prevents yielding empty/overlapped segments.
    """

    def __init__(self, overlap_seconds: float = 0.0, sample_rate: int = 16000):
        self.overlap_seconds = overlap_seconds
        self.sample_rate = sample_rate
        self.overlap_samples = int(overlap_seconds * sample_rate)
        self.last_yielded_end_sample: int = 0
        self.prev_segment: Optional[SpeechSegment] = None
        self.curr_segment: Optional[SpeechSegment] = None

    def update(self, speech_ts: List[SpeechSegment]) -> Optional[SpeechSegment]:
        """Update with latest speech timestamps and return completed segment if a new one started.

        Args:
            speech_ts: Latest list of speech segments from VAD.

        Returns:
            Completed previous segment (with overlap applied) or None if no completion.
        """
        if not speech_ts:
            return None

        new_curr = speech_ts[-1]
        new_prev = speech_ts[-2] if len(speech_ts) > 1 else None

        completed = None
        # Detect if a new segment has started: current exists and its start differs from new current
        if (
            self.curr_segment is not None
            and new_curr["start"] != self.curr_segment["start"]
            and self.prev_segment is not None
        ):
            # A new segment started → the previous one (second-to-last) is now complete
            completed = self._prepare_completed_segment(self.prev_segment)

        # Update state after checking
        self.curr_segment = new_curr
        self.prev_segment = new_prev

        return completed

    def get_final_segment(self) -> Optional[SpeechSegment]:
        """Return the last pending segment at end of recording (if any)."""
        # At end of stream, the last completed segment is the previous one
        # (the final current segment becomes "previous" when no more updates)
        if self.prev_segment is not None:
            return self._prepare_completed_segment(self.prev_segment)
        return None

    def _prepare_completed_segment(self, segment: SpeechSegment) -> Optional[SpeechSegment]:
        """Apply overlap and return segment ready for yield, or None if it would be empty."""
        original_start = segment["start"]
        effective_start = max(self.last_yielded_end_sample, int(segment["start"]) - self.overlap_samples)

        if effective_start >= int(segment["end"]):
            # Overlap consumed the entire segment
            self.last_yielded_end_sample = int(segment["end"])
            return None

        # Modify copy for yielding
        yield_segment = segment.copy()
        yield_segment["start"] = effective_start
        yield_segment["duration"] = (yield_segment["end"] - yield_segment["start"]) / self.sample_rate

        self.last_yielded_end_sample = int(segment["end"])
        return yield_segment
