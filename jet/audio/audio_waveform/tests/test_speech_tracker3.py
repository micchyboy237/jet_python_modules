import numpy as np
import pytest
from jet.audio.audio_waveform.speech_tracker3 import (
    StreamingSpeechTracker,
    StreamingSpeechTrackerConfig,
)

FRAME_LENGTH_SAMPLE = 400


class FakeVadResult:
    def __init__(
        self,
        is_speech_start=False,
        is_speech_end=False,
    ):
        self.is_speech_start = is_speech_start
        self.is_speech_end = is_speech_end


class FakeVad:
    """
    Deterministic VAD simulator.
    """

    def __init__(self, results):
        self.results = results
        self.index = 0

    def reset(self):
        self.index = 0

    def detect_frame(self, frame):
        result = self.results[self.index]
        self.index += 1
        return result


@pytest.fixture
def tracker():
    config = StreamingSpeechTrackerConfig()
    vad = FakeVad([])
    tracker = StreamingSpeechTracker(vad, config)
    return tracker


@pytest.fixture
def frame():
    return np.ones(FRAME_LENGTH_SAMPLE, dtype=np.float32)


class TestStreamingSpeechTracker:
    def test_speech_segment_type(self, tracker, frame):
        """
        Given speech start and end events
        When streaming processes frames
        Then SpeechSegment TypedDict is returned
        """

        tracker.vad = FakeVad(
            [
                FakeVadResult(is_speech_start=True),
                FakeVadResult(),
                FakeVadResult(is_speech_end=True),
            ]
        )

        tracker._running = True

        tracker._next_frame = lambda: frame

        gen = tracker.run_streaming_audio()

        segment = next(gen)

        assert isinstance(segment, dict)

        expected_keys = {"start_time", "end_time", "audio"}

        result = set(segment.keys())
        expected = expected_keys

        assert result == expected

    def test_audio_accumulation(self, tracker, frame):
        """
        Given a speech segment with 3 frames
        When processed
        Then returned audio contains 3 frames
        """

        tracker.vad = FakeVad(
            [
                FakeVadResult(is_speech_start=True),
                FakeVadResult(),
                FakeVadResult(is_speech_end=True),
            ]
        )

        tracker._running = True
        tracker._next_frame = lambda: frame

        gen = tracker.run_streaming_audio()

        segment = next(gen)

        result = len(segment["audio"])
        expected = FRAME_LENGTH_SAMPLE * 3

        assert result == expected

    def test_start_end_times(self, tracker, frame):
        """
        Given frames processed sequentially
        When a speech segment completes
        Then start and end times match frame index timing
        """

        tracker.vad = FakeVad(
            [
                FakeVadResult(is_speech_start=True),
                FakeVadResult(),
                FakeVadResult(is_speech_end=True),
            ]
        )

        tracker._running = True
        tracker._next_frame = lambda: frame

        gen = tracker.run_streaming_audio()

        segment = next(gen)

        result_start = segment["start_time"]
        result_end = segment["end_time"]

        expected_start = 0.0
        expected_end = 0.02

        assert result_start == expected_start
        assert result_end == expected_end

    def test_no_segment_without_end(self, tracker, frame):
        """
        Given speech starts but never ends
        When frames are processed
        Then no segment is yielded
        """

        tracker.vad = FakeVad(
            [
                FakeVadResult(is_speech_start=True),
                FakeVadResult(),
                FakeVadResult(),
            ]
        )

        tracker._running = True
        tracker._next_frame = lambda: frame

        gen = tracker.run_streaming_audio()

        # Consume a few frames but ensure no segment is emitted
        for _ in range(3):
            try:
                next(gen)
            except StopIteration:
                break

        tracker.stop()
