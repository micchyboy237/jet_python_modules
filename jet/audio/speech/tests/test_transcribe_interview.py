import pytest
from unittest.mock import Mock
import speech_recognition as sr
from jet.audio.speech.transcribe_interview import Transcriber
from typing import Optional


@pytest.fixture
def transcriber():
    """Fixture to create a Transcriber instance with a mocked recognizer."""
    recognizer = Mock(spec=sr.Recognizer)
    return Transcriber(recognizer=recognizer)


class TestTranscriber:
    """Test suite for the Transcriber class."""

    def test_transcribe_audio_success(self, transcriber):
        """Test successful transcription of audio."""
        # Given: A valid audio input and mocked recognizer
        audio = Mock(spec=sr.AudioData)
        expected_transcription = "Hello, this is a test"
        transcriber.recognizer.recognize_google.return_value = expected_transcription

        # When: Transcribing the audio
        result = transcriber.transcribe_audio(audio)

        # Then: The transcription should match the expected output
        assert result == expected_transcription, f"Expected '{expected_transcription}', but got '{result}'"

    def test_transcribe_audio_unknown_value_error(self, transcriber):
        """Test handling of unrecognizable audio."""
        # Given: An audio input that cannot be understood
        audio = Mock(spec=sr.AudioData)
        transcriber.recognizer.recognize_google.side_effect = sr.UnknownValueError()

        # When: Transcribing the audio
        result = transcriber.transcribe_audio(audio)

        # Then: The result should be None
        expected = None
        assert result == expected, f"Expected None, but got '{result}'"

    def test_transcribe_audio_request_error(self, transcriber):
        """Test handling of API request failure."""
        # Given: An audio input with a mocked API failure
        audio = Mock(spec=sr.AudioData)
        transcriber.recognizer.recognize_google.side_effect = sr.RequestError(
            "API failure")

        # When: Transcribing the audio
        result = transcriber.transcribe_audio(audio)

        # Then: The result should be None
        expected = None
        assert result == expected, f"Expected None, but got '{result}'"

    def test_adjust_for_ambient_noise_success(self, transcriber):
        """Test successful ambient noise adjustment."""
        # Given: A transcriber with a mocked microphone and recognizer
        transcriber.microphone = Mock(spec=sr.Microphone)
        transcriber.recognizer.adjust_for_ambient_noise = Mock()

        # When: Adjusting for ambient noise
        transcriber.adjust_for_ambient_noise(duration=1.0)

        # Then: The adjust_for_ambient_noise method should be called
        transcriber.recognizer.adjust_for_ambient_noise.assert_called_once()

    def test_adjust_for_ambient_noise_failure(self, transcriber):
        """Test handling of ambient noise adjustment failure."""
        # Given: A transcriber with a mocked microphone and failing recognizer
        transcriber.microphone = Mock(spec=sr.Microphone)
        transcriber.recognizer.adjust_for_ambient_noise.side_effect = Exception(
            "Microphone error")

        # When: Adjusting for ambient noise
        # Then: An exception should be raised
        with pytest.raises(Exception, match="Microphone error"):
            transcriber.adjust_for_ambient_noise()
