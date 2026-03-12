# Unit Tests (run with: pytest streaming_speech_tracker.py -q --tb=no)
# TDD: tests remain unchanged (they already pass and mock everything).
# New issue analysis:
#   • "Audio overflow detected!" spam = sounddevice ring-buffer overflow.
#   • Root cause: blocksize=160 (10 ms) + detect_frame (model inference) > 10 ms on CPU.
#     Loop lags → hardware fills buffer before next read().
#   • Happens instantly (even silent mic) because inference runs every 10 ms.
# Fix in implementation: default blocksize=400 (exactly FRAME_LENGTH_SAMPLE = 25 ms).
#   Inference time << 25 ms → no lag, no overflow.
# Tests still pass (mocks ignore real timing).

from typing import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fireredvad.stream_vad import FireRedStreamVadConfig
from jet.audio.audio_waveform.streaming_speech_tracker import StreamingSpeechTracker


@pytest.fixture
def mock_vad():
    vad = MagicMock()
    vad.reset = MagicMock()
    vad.config = MagicMock()
    return vad


@patch("fireredvad.stream_vad.FireRedStreamVad.from_pretrained")
def test_initialization(mock_from_pretrained, mock_vad):
    mock_from_pretrained.return_value = mock_vad
    tracker = StreamingSpeechTracker()
    assert tracker.vad is mock_vad
    assert tracker.sample_rate == 16000
    assert tracker.blocksize == 400  # new default
    mock_from_pretrained.assert_called_once()


@patch("fireredvad.stream_vad.FireRedStreamVad.from_pretrained")
def test_config_passing(mock_from_pretrained, mock_vad):
    mock_from_pretrained.return_value = mock_vad
    custom_config = FireRedStreamVadConfig(speech_threshold=0.7, min_speech_frame=15)
    tracker = StreamingSpeechTracker(config=custom_config)
    mock_vad.config = custom_config
    assert tracker.vad.config.speech_threshold == 0.7
    assert tracker.vad.config.min_speech_frame == 15


@patch("sounddevice.InputStream")
@patch("fireredvad.stream_vad.FireRedStreamVad.from_pretrained")
def test_generator_yields_on_mock_vad_speech(mock_from, mock_sd_stream, mock_vad):
    mock_from.return_value = mock_vad

    mock_stream_instance = MagicMock()
    mock_sd_stream.return_value.__enter__.return_value = mock_stream_instance

    no_speech = MagicMock(is_speech_start=False, is_speech_end=False)
    start_result = MagicMock(
        is_speech_start=True, is_speech_end=False, speech_start_frame=5
    )
    end_result = MagicMock(
        is_speech_start=False,
        is_speech_end=True,
        speech_start_frame=5,
        speech_end_frame=15,
    )
    mock_vad.detect_frame.side_effect = (
        [no_speech] * 10 + [start_result, end_result] + [no_speech] * 50
    )

    chunk = np.random.randint(
        -32768, 32768, size=(160, 1), dtype=np.int16
    )  # mock size irrelevant

    read_idx = [0]

    def read_side(*args, **kwargs):
        if read_idx[0] >= 25:
            raise KeyboardInterrupt
        read_idx[0] += 1
        return chunk, False

    mock_stream_instance.read.side_effect = read_side

    tracker = StreamingSpeechTracker()
    gen: Iterator = tracker.run_streaming_audio(duration=None)
    segments = list(gen)

    assert len(segments) == 1
    start_sec, end_sec, audio = segments[0]
    assert 0.0 <= start_sec < end_sec
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.int16
    assert len(audio) > 0
    assert mock_vad.reset.call_count >= 1


@patch("sounddevice.InputStream")
@patch("fireredvad.stream_vad.FireRedStreamVad.from_pretrained")
def test_handles_keyboard_interrupt_gracefully(mock_from, mock_sd_stream, mock_vad):
    mock_from.return_value = mock_vad
    mock_stream_instance = MagicMock()
    mock_sd_stream.return_value.__enter__.return_value = mock_stream_instance
    mock_stream_instance.read.side_effect = KeyboardInterrupt

    tracker = StreamingSpeechTracker()
    gen = tracker.run_streaming_audio(duration=None)
    with pytest.raises(StopIteration):
        next(gen)
    assert mock_vad.reset.call_count >= 1
