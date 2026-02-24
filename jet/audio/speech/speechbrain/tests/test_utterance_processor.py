"""
Pytest unit tests for process_utterance_buffer and StreamingSpeechProcessor.

- BDD-style/structured with clear naming
- Uses unittest.mock.patch for speech_timestamps_extractor
- Covers empty, silence-only, speech-completed, ongoing speech, chunk-completing
- Separate classes for stateless and streaming cases
- Ensures buffer, output payload, inner state as expected
"""

from unittest.mock import patch

import numpy as np
import pytest
from jet.audio.speech.speechbrain.utterance_processor import (
    StreamingSpeechProcessor,
    process_utterance_buffer,
)

SR = 16000


def _make_speech_segment(num, start, end, prob=0.92):
    """Create a canonical speech segment as returned by the extractor."""
    return {
        "num": num,
        "start": start,
        "end": end,
        "prob": prob,
        "duration": round(end - start, 3),
        "frames_length": int((end - start) * 100),  # fake but consistent
        "frame_start": int(start * 100),
        "frame_end": int(end * 100),
        "type": "speech",
        "segment_probs": [],
    }


class TestProcessUtteranceBuffer:
    """
    Unit tests for the process_utterance_buffer function.
    These tests examine stateless buffer input/output logic.
    """

    @patch(
        "jet.audio.speech.speechbrain.speech_timestamps_extractor.extract_speech_timestamps"
    )
    def test_empty_buffer_no_chunk(self, mock_extract):
        # Given
        buffer = np.array([], dtype=np.float32)
        mock_extract.return_value = []

        # When
        result_submitted, result_buffer, result_payload = process_utterance_buffer(
            utterance_audio_buffer=buffer,
            new_audio_chunk=None,
            sampling_rate=SR,
            context_sec=1.0,
            min_silence_for_completion_sec=0.35,
        )

        # Then
        assert result_submitted == []
        assert result_buffer.shape == (0,)
        assert result_payload == {
            "speech_segments": [],
            "total_buffer_duration": 0.0,
            "submitted_count": 0,
        }

    @patch(
        "jet.audio.speech.speechbrain.speech_timestamps_extractor.extract_speech_timestamps"
    )
    def test_only_silence_buffer_retains_tail(self, mock_extract):
        # Given
        buffer = np.zeros(int(3.0 * SR), dtype=np.float32)
        mock_extract.return_value = []

        # When
        result_submitted, result_buffer, result_payload = process_utterance_buffer(
            utterance_audio_buffer=buffer,
            new_audio_chunk=None,
            sampling_rate=SR,
            context_sec=1.0,
        )

        # Then
        expected_buffer = buffer[-int(1.0 * SR) :]
        assert result_submitted == []
        assert np.array_equal(result_buffer, expected_buffer)
        assert result_payload == {
            "speech_segments": [],
            "total_buffer_duration": 3.0,
            "submitted_count": 0,
        }

    @patch(
        "jet.audio.speech.speechbrain.speech_timestamps_extractor.extract_speech_timestamps"
    )
    def test_speech_completed_by_silence(self, mock_extract):
        # Given: 2s silence, 2.5s speech, 0.5s silence. Speech boundary at 2.0-4.5s
        silence1 = np.zeros(int(2.0 * SR), dtype=np.float32)
        speech = np.sin(np.linspace(0, 2.5 * 2 * np.pi, int(2.5 * SR))).astype(
            np.float32
        )
        silence2 = np.zeros(int(0.5 * SR), dtype=np.float32)
        full = np.concatenate([silence1, speech, silence2])

        # Fake: 1 speech segment, 1 dummy non-speech
        mock_extract.return_value = [
            _make_speech_segment(1, 2.0, 4.5),
            {
                "num": 2,
                "start": 4.5,
                "end": 5.0,
                "type": "non-speech",
                "dummy": True,
            },
        ]

        # When
        result_submitted, result_buffer, result_payload = process_utterance_buffer(
            utterance_audio_buffer=full,
            new_audio_chunk=None,
            sampling_rate=SR,
            min_silence_for_completion_sec=0.35,
        )

        # Then
        expected = [_make_speech_segment(1, 2.0, 4.5)]
        expected_buffer = full[int(4.5 * SR) :]
        assert result_submitted == expected
        assert np.array_equal(result_buffer, expected_buffer)
        assert result_payload == {
            "speech_segments": expected,
            "total_buffer_duration": 5.0,
            "submitted_count": 1,
        }

    @patch(
        "jet.audio.speech.speechbrain.speech_timestamps_extractor.extract_speech_timestamps"
    )
    def test_ongoing_speech_no_completion(self, mock_extract):
        # Given: 4s of audio, ongoing speech segment
        full = np.zeros(int(4.0 * SR), dtype=np.float32)
        mock_extract.return_value = [_make_speech_segment(1, 1.0, 3.8)]  # ongoing

        # When
        result_submitted, result_buffer, result_payload = process_utterance_buffer(
            utterance_audio_buffer=full,
            new_audio_chunk=None,
            sampling_rate=SR,
            min_silence_for_completion_sec=0.35,
        )

        # Then
        assert result_submitted == []
        assert np.array_equal(result_buffer, full)
        assert result_payload == {
            "speech_segments": [],
            "total_buffer_duration": 4.0,
            "submitted_count": 0,
        }

    @patch(
        "jet.audio.speech.speechbrain.speech_timestamps_extractor.extract_speech_timestamps"
    )
    def test_new_chunk_completes_segment(self, mock_extract):
        # Given: previous ongoing + new silence triggers commit
        prev = np.zeros(int(3.5 * SR), dtype=np.float32)
        new_chunk = np.zeros(int(0.6 * SR), dtype=np.float32)
        mock_extract.return_value = [_make_speech_segment(1, 1.0, 3.5)]

        # When
        result_submitted, result_buffer, result_payload = process_utterance_buffer(
            utterance_audio_buffer=prev,
            new_audio_chunk=new_chunk,
            sampling_rate=SR,
            min_silence_for_completion_sec=0.35,
        )

        # Then
        expected = [_make_speech_segment(1, 1.0, 3.5)]
        expected_buffer = np.concatenate([prev[int(3.5 * SR) :], new_chunk])
        assert result_submitted == expected
        assert np.array_equal(result_buffer, expected_buffer)
        assert result_payload == {
            "speech_segments": expected,
            "total_buffer_duration": 4.1,
            "submitted_count": 1,
        }


class TestStreamingSpeechProcessor:
    """
    Unit tests for StreamingSpeechProcessor (stateful streaming logic)
    """

    @patch(
        "jet.audio.speech.speechbrain.speech_timestamps_extractor.extract_speech_timestamps"
    )
    def test_silence_buffer_grows_and_is_capped(self, mock_extract):
        # Given
        processor = StreamingSpeechProcessor(
            context_sec=1.0,
            min_silence_for_completion_sec=0.35,
        )
        chunk = np.zeros(int(0.1 * SR), dtype=np.float32)
        mock_extract.return_value = []

        # When (simulate 1.5s silence fed as 15 chunks)
        for _ in range(15):
            payload = processor.process(chunk)

        # Then
        result_buffer = processor.utterance_audio_buffer
        assert len(payload["speech_segments"]) == 0
        assert payload["submitted_count"] == 0
        assert len(result_buffer) == int(1.0 * SR)  # max context cap

    @patch(
        "jet.audio.speech.speechbrain.speech_timestamps_extractor.extract_speech_timestamps"
    )
    def test_speech_committed_buffer_trimmed(self, mock_extract):
        # Given: 1s silence, 2.5s speech, 0.5s silence
        processor = StreamingSpeechProcessor(context_sec=1.0)
        full = np.concatenate(
            [
                np.zeros(int(1.0 * SR), dtype=np.float32),
                np.sin(np.linspace(0, 2.5 * 2 * np.pi, int(2.5 * SR))).astype(
                    np.float32
                ),
                np.zeros(int(0.5 * SR), dtype=np.float32),
            ]
        )
        mock_extract.return_value = [_make_speech_segment(1, 1.0, 3.5)]

        # When
        payload = processor.process(full)

        # Then
        expected = [_make_speech_segment(1, 1.0, 3.5)]
        expected_buffer = full[int(3.5 * SR) :]
        assert payload["speech_segments"] == expected
        assert payload["submitted_count"] == 1
        assert np.array_equal(processor.utterance_audio_buffer, expected_buffer)

    @patch(
        "jet.audio.speech.speechbrain.speech_timestamps_extractor.extract_speech_timestamps"
    )
    def test_reset_clears_buffer(self, mock_extract):
        processor = StreamingSpeechProcessor()
        processor.process(np.zeros(int(2.0 * SR), dtype=np.float32))
        mock_extract.return_value = []

        # When
        processor.reset()
        # Then
        assert np.array_equal(
            processor.utterance_audio_buffer, np.array([], dtype=np.float32)
        )


if __name__ == "__main__":
    pytest.main(["-q", __file__])
