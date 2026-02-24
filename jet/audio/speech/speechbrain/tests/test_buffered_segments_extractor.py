from typing import Any, Dict, List
from unittest.mock import Mock, patch

import jet.audio.speech.speechbrain.speech_timestamps_extractor as speech_extractor
import numpy as np
import pytest
from jet.audio.speech.speechbrain.buffered_segments_extractor import (
    extract_buffered_segments,
)

# Minimal SpeechSegment for tests
SpeechSegmentDict = Dict[str, Any]


@pytest.fixture
def sample_buffer_with_trailing_silence() -> bytearray:
    """2 s buffer (silence-speech-silence) – real-world streaming example."""
    sr = 16000
    total_samples = int(2.0 * sr)
    audio_np = np.zeros(total_samples, dtype=np.int16)
    speech_start = int(0.5 * sr)
    speech_end = int(1.5 * sr)
    audio_np[speech_start:speech_end] = 5000
    return bytearray(audio_np.tobytes())


class TestExtractBufferedSegments:
    """Tests for extract_buffered_segments following BDD principles.

    All tests use mocks for fast, deterministic execution.
    """

    @patch.object(speech_extractor, "extract_speech_timestamps")
    def test_given_all_non_speech_buffer_when_extract_buffered_segments_then_returns_empty_joined_and_full_trailing(
        self, mock_extract: Mock
    ):
        # Given: Pure silence buffer detected as single non-speech segment (common start-of-stream case)
        sr = 16000
        buffer = bytearray(
            b"\x00\x00" * (sr * 1)
        )  # 1 s = 32 kB – keeps failure output readable
        mock_segments: List[SpeechSegmentDict] = [
            {
                "num": 1,
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "type": "non-speech",
                "prob": 0.05,
            }
        ]
        expected_combined = bytearray()
        expected_trailing = buffer[:]  # full buffer is the trailing silence

        # When
        mock_extract.return_value = mock_segments
        result_combined, result_trailing = extract_buffered_segments(buffer)

        # Then
        mock_extract.assert_called_once()
        assert result_combined == expected_combined
        assert result_trailing == expected_trailing
        assert result_combined + (result_trailing or bytearray()) == buffer

    @patch.object(speech_extractor, "extract_speech_timestamps")
    def test_given_buffer_ends_with_speech_when_extract_buffered_segments_then_joined_is_full_trailing_is_none(
        self, mock_extract: Mock, sample_buffer_with_trailing_silence: bytearray
    ):
        # Given: Buffer ends with speech segment (no trailing silence to buffer)
        mock_segments: List[SpeechSegmentDict] = [
            {
                "num": 1,
                "start": 0.0,
                "end": 2.0,
                "duration": 2.0,
                "type": "speech",
                "prob": 0.92,
            }
        ]
        expected_combined = sample_buffer_with_trailing_silence[:]
        expected_trailing: bytearray | None = None

        # When
        mock_extract.return_value = mock_segments
        result_combined, result_trailing = extract_buffered_segments(
            sample_buffer_with_trailing_silence
        )

        # Then
        mock_extract.assert_called_once()
        assert result_combined == expected_combined
        assert result_trailing == expected_trailing

    @patch.object(speech_extractor, "extract_speech_timestamps")
    def test_given_trailing_non_speech_when_extract_buffered_segments_then_joins_all_except_last_trailing_returned_separately(
        self, mock_extract: Mock, sample_buffer_with_trailing_silence: bytearray
    ):
        # Given: Real-world streaming buffer - speech followed by trailing silence (last non-speech)
        mock_segments: List[SpeechSegmentDict] = [
            {
                "num": 1,
                "start": 0.0,
                "end": 0.5,
                "duration": 0.5,
                "type": "non-speech",
                "prob": 0.08,
            },
            {
                "num": 2,
                "start": 0.5,
                "end": 1.5,
                "duration": 1.0,
                "type": "speech",
                "prob": 0.95,
            },
            {
                "num": 3,
                "start": 1.5,
                "end": 2.0,
                "duration": 0.5,
                "type": "non-speech",
                "prob": 0.07,
            },
        ]
        expected_combined = sample_buffer_with_trailing_silence[
            0 : int(1.5 * 16000) * 2
        ]
        expected_trailing = sample_buffer_with_trailing_silence[int(1.5 * 16000) * 2 :]

        # When
        mock_extract.return_value = mock_segments
        result_combined, result_trailing = extract_buffered_segments(
            sample_buffer_with_trailing_silence
        )

        # Then
        mock_extract.assert_called_once()
        assert result_combined == expected_combined
        assert result_trailing == expected_trailing
        assert (
            result_combined + (result_trailing or bytearray())
            == sample_buffer_with_trailing_silence
        )

    @patch.object(speech_extractor, "extract_speech_timestamps")
    def test_given_empty_buffer_when_extract_buffered_segments_then_returns_empty_combined_and_none(
        self, mock_extract: Mock
    ):
        # Given: No audio yet (edge case at start of recording)
        buffer = bytearray()
        expected_combined = bytearray()
        expected_trailing: bytearray | None = None

        # When
        mock_extract.return_value = []
        result_combined, result_trailing = extract_buffered_segments(buffer)

        # Then
        assert result_combined == expected_combined
        assert result_trailing == expected_trailing
        mock_extract.assert_not_called()  # early return → VAD never called

    @patch.object(speech_extractor, "extract_speech_timestamps")
    def test_given_is_partial_flag_when_extract_buffered_segments_then_logic_unchanged(
        self, mock_extract: Mock, sample_buffer_with_trailing_silence: bytearray
    ):
        # Given: Same trailing case but is_partial=True (flag kept for compatibility)
        mock_segments: List[SpeechSegmentDict] = [
            {
                "num": 1,
                "start": 0.0,
                "end": 2.0,
                "duration": 2.0,
                "type": "speech",
                "prob": 0.9,
            }
        ]
        expected_combined = sample_buffer_with_trailing_silence[:]
        expected_trailing: bytearray | None = None

        # When
        mock_extract.return_value = mock_segments
        result_combined, result_trailing = extract_buffered_segments(
            sample_buffer_with_trailing_silence, is_partial=True
        )

        # Then
        mock_extract.assert_called_once()
        assert result_combined == expected_combined
        assert result_trailing == expected_trailing
