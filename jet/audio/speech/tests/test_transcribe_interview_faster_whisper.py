import pytest
import numpy as np
from unittest.mock import Mock, patch
from jet.audio.speech.transcribe_interview_faster_whisper import Transcriber
from typing import Optional, Literal


@pytest.fixture
def transcriber():
    """Fixture to create a Transcriber instance with a mocked Faster-Whisper model."""
    with patch("faster_whisper.WhisperModel") as mock_whisper:
        mock_model = Mock()
        mock_whisper.return_value = mock_model
        transcriber = Transcriber(model_type="base")
        transcriber.pyaudio_instance = Mock()  # Mock PyAudio to avoid real audio
        return transcriber


class TestTranscriber:
    """Test suite for the Transcriber class using Faster-Whisper."""

    def test_transcribe_audio_success(self, transcriber):
        """Test successful transcription of audio."""
        # Given: A valid audio input and mocked Faster-Whisper model
        audio = np.zeros(16000 * 5, dtype=np.float32)  # 5 seconds of silence
        expected_transcription = "Hello, this is a test interview"
        mock_segment = Mock()
        mock_segment.text = expected_transcription
        transcriber.model.transcribe.return_value = ([mock_segment], {})

        # When: Transcribing the audio
        with patch("wave.open") as mock_wave, patch("os.unlink"):
            result = transcriber.transcribe_audio(audio)

        # Then: The transcription should match the expected output
        assert result == expected_transcription, f"Expected '{expected_transcription}', but got '{result}'"

    def test_transcribe_audio_failure(self, transcriber):
        """Test handling of transcription failure."""
        # Given: An audio input with a mocked Faster-Whisper failure
        audio = np.zeros(16000 * 5, dtype=np.float32)
        transcriber.model.transcribe.side_effect = Exception("Whisper error")

        # When: Transcribing the audio
        with patch("wave.open") as mock_wave, patch("os.unlink"):
            result = transcriber.transcribe_audio(audio)

        # Then: The result should be None
        expected = None
        assert result == expected, f"Expected None, but got '{result}'"

    def test_start_audio_stream(self, transcriber):
        """Test audio stream starts and stops correctly."""
        # Given: A transcriber with a mocked PyAudio instance
        mock_stream = Mock()
        transcriber.pyaudio_instance.open.return_value = mock_stream
        transcriber.is_running = True

        # When: Starting the audio stream
        with patch.object(transcriber, "start_audio_stream") as mock_start:
            transcriber.start_audio_stream()
            mock_start.assert_called_once()

        # Then: The stream should be opened and closed
        transcriber.pyaudio_instance.open.assert_called_once()
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_stop_transcription(self, transcriber):
        """Test stopping the transcription process."""
        # Given: A running transcriber with a mocked thread
        transcriber.is_running = True
        transcriber.audio_thread = Mock()
        transcriber.pyaudio_instance.terminate = Mock()

        # When: Stopping transcription
        transcriber.stop_transcription()

        # Then: The thread should be joined and PyAudio terminated
        assert transcriber.is_running == False, "Expected is_running to be False"
        transcriber.audio_thread.join.assert_called_once()
        transcriber.pyaudio_instance.terminate.assert_called_once()
