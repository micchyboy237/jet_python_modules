import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Optional
from jet.audio.system.transcribe_system_audio import AudioTranscriber


@pytest.fixture
def transcriber():
    """Fixture to create a transcriber instance with mocked Whisper model."""
    transcriber = AudioTranscriber(model_size="tiny")
    transcriber.model = Mock()
    return transcriber


class TestAudioTranscriber:
    @pytest.mark.asyncio
    async def test_speech_detection_and_transcription(self, transcriber):
        audio_data = np.array([1000, -1000, 500, -500], dtype=np.int16)
        expected_transcription = "Hello world"
        transcriber.model.transcribe.return_value = (
            [Mock(text="Hello"), Mock(text="world")], None)

        with patch('sounddevice.InputStream', autospec=True) as mock_stream:
            mock_stream.side_effect = lambda **kwargs: Mock(
                __enter__=lambda x: x, __exit__=lambda *args: None)
            transcriber.frames = [audio_data]
            result = await transcriber.capture_and_transcribe()

        expected = expected_transcription
        assert result == expected, f"Expected transcription '{expected}', got '{result}'"
        transcriber.model.transcribe.assert_called_with(
            np.concatenate([audio_data]).astype(np.float32) / 32768.0,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            log_progress=True
        )

    def test_callback_appends_audio(self, transcriber):
        audio_data = np.array([1000, -1000, 500, -500], dtype=np.int16)

        transcriber.frames = []
        transcriber.callback(audio_data, None, Mock(),
                             Mock(spec=sd.CallbackFlags))

        assert len(transcriber.frames) == 1, "Expected one audio frame"
        np.testing.assert_array_equal(transcriber.frames[0], audio_data)

    @pytest.mark.asyncio
    async def test_async_transcription(self, transcriber):
        expected_transcription = "Test speech"
        transcriber.model.transcribe.return_value = (
            [Mock(text="Test"), Mock(text="speech")], None)
        audio_data = np.array([1000, -1000, 500, -500], dtype=np.int16)

        with patch('sounddevice.InputStream', autospec=True) as mock_stream:
            mock_stream.side_effect = lambda **kwargs: Mock(
                __enter__=lambda x: x, __exit__=lambda *args: None)
            transcriber.frames = [audio_data]
            result = await transcriber.capture_and_transcribe()

        expected = expected_transcription
        assert result == expected, f"Expected transcription '{expected}', got '{result}'"
        transcriber.model.transcribe.assert_called_with(
            np.concatenate([audio_data]).astype(np.float32) / 32768.0,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            log_progress=True
        )


@pytest.mark.asyncio
async def test_async_transcription(transcriber):
    """Test async transcription workflow."""
    # Given: Mock transcription result
    expected_transcription = "Test speech"
    transcriber.model.transcribe.return_value = (
        [Mock(text="Test"), Mock(text="speech")], None)
    audio_data = np.array([1000, -1000, 500, -500], dtype=np.int16)

    # When: Simulate async capture and transcription
    with patch.object(transcriber, 'frames', [audio_data]):
        result = await transcriber.capture_and_transcribe()

    # Then: Verify transcription is correct
    expected = expected_transcription
    assert result == expected, f"Expected transcription '{expected}', got '{result}'"


def teardown_method(self):
    """Clean up after each test."""
    pass
