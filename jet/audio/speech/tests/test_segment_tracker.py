import pytest

from typing import List

from jet.audio.speech.silero.speech_types import SpeechSegment

from jet.audio.speech.segment_tracker import SpeechSegmentTracker


@pytest.fixture
def sample_segments() -> List[SpeechSegment]:
    # Real-world style segments (sample-based, 16000 Hz)
    return [
        {"num": 1, "start": 0, "end": 32000, "prob": 0.85, "duration": 2.0, "frames_length": 62, "frame_start": 0, "frame_end": 61, "segment_probs": []},
        {"num": 2, "start": 48000, "end": 80000, "prob": 0.90, "duration": 2.0, "frames_length": 62, "frame_start": 93, "frame_end": 155, "segment_probs": []},
        {"num": 3, "start": 96000, "end": 128000, "prob": 0.88, "duration": 2.0, "frames_length": 62, "frame_start": 187, "frame_end": 249, "segment_probs": []},
    ]


def test_no_segments_returns_none(sample_segments):
    # Given: A tracker with no overlap
    tracker = SpeechSegmentTracker(overlap_seconds=0.0)

    # When: Updating with empty list
    result = tracker.update([])

    # Then: No segment is completed
    expected = None
    assert result == expected


def test_single_segment_yielded_at_end(sample_segments):
    # Given: Tracker with no overlap
    tracker = SpeechSegmentTracker(overlap_seconds=0.0)
    segment = sample_segments[0]

    # When: Updating with one segment
    result_during = tracker.update([segment])

    # Then: Nothing yielded during recording
    expected_during = None
    assert result_during == expected_during

    # When: Requesting final segment
    result_final = tracker.get_final_segment()

    # Then: The segment is returned unchanged
    expected_final = segment.copy()
    expected_final["duration"] = 2.0  # unchanged
    assert result_final == expected_final


def test_multiple_segments_yield_previous_when_new_appears(sample_segments):
    # Given: Tracker with no overlap
    tracker = SpeechSegmentTracker(overlap_seconds=0.0)

    # When: First update with segment 1
    result1 = tracker.update(sample_segments[:1])

    # Then: Nothing yielded yet
    expected1 = None
    assert result1 == expected1

    # When: Second update with segments 1 and 2
    result2 = tracker.update(sample_segments[:2])

    # Then: Segment 1 is completed and yielded
    expected2 = sample_segments[0].copy()
    expected2["duration"] = 2.0
    assert result2 == expected2

    # When: Third update with all segments
    result3 = tracker.update(sample_segments)

    # Then: Segment 2 is now completed
    expected3 = sample_segments[1].copy()
    expected3["duration"] = 2.0
    assert result3 == expected3

    # When: Final segment requested
    final = tracker.get_final_segment()

    # Then: Segment 3 is yielded
    expected_final = sample_segments[2].copy()
    expected_final["duration"] = 2.0
    assert final == expected_final


def test_overlap_applied_correctly():
    # Given: Tracker with 0.5s overlap (8000 samples at 16kHz)
    tracker = SpeechSegmentTracker(overlap_seconds=0.5)

    seg1 = {"num": 1, "start": 16000, "end": 48000, "prob": 0.9, "duration": 2.0, "frames_length": 62, "frame_start": 31, "frame_end": 93, "segment_probs": []}
    seg2 = {"num": 2, "start": 64000, "end": 96000, "prob": 0.9, "duration": 2.0, "frames_length": 62, "frame_start": 125, "frame_end": 187, "segment_probs": []}

    # When: Update with both segments
    completed = tracker.update([seg1, seg2])

    # Then: First segment has overlap applied → start moved back by 8000 samples
    expected = seg1.copy()
    expected["start"] = 48000 - 8000  # 40000
    expected["duration"] = (48000 - 40000) / 16000  # 0.5s
    assert completed == expected


def test_overlap_skips_empty_segment():
    # Given: Tracker with large overlap (3s = 48000 samples)
    tracker = SpeechSegmentTracker(overlap_seconds=3.0)

    seg1 = {"num": 1, "start": 0, "end": 32000, "prob": 0.9, "duration": 2.0, "frames_length": 62, "frame_start": 0, "frame_end": 61, "segment_probs": []}
    seg2 = {"num": 2, "start": 33600, "end": 65600, "prob": 0.9, "duration": 2.0, "frames_length": 62, "frame_start": 65, "frame_end": 128, "segment_probs": []}

    # When: Update with both → seg1 end=32000, overlap wants start=32000-48000=-16000 → effective start=32000 → empty
    completed = tracker.update([seg1, seg2])

    # Then: No segment yielded (skipped), but last_yielded_end_sample updated to seg1.end
    expected = None
    assert completed == expected

    # And final segment should still be yieldable (seg2)
    final = tracker.get_final_segment()
    expected_final = seg2.copy()
    expected_final["start"] = max(tracker.last_yielded_end_sample, int(seg2["start"]) - 48000)
    expected_final["duration"] = (seg2["end"] - expected_final["start"]) / 16000
    assert final == expected_final

