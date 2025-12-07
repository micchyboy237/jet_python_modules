import pytest
import numpy as np
from jet.audio.speech.pyannote.speech_speakers_extractor import extract_speech_speakers, SpeechSpeakerSegment

@pytest.fixture
def mock_audio_array():
    return np.random.rand(16000 * 5).astype(np.float32)  # 5s mono audio

def test_extract_speech_speakers_array(mocker, mock_audio_array):
    # Given: Mock pipeline to return a fixed output
    mock_pipeline = mocker.patch("pyannote.audio.Pipeline.from_pretrained")
    mock_output = mocker.Mock()
    mock_output.itertracks.return_value = [
        (mocker.Mock(start=0.0, end=1.0), None, "SPEAKER_00"),
        (mocker.Mock(start=2.0, end=3.0), None, "SPEAKER_01"),
    ]
    mock_pipeline.return_value.__call__.return_value = mock_output
    token = "fake_token"

    # When
    result = extract_speech_speakers(mock_audio_array, token=token)

    # Then
    expected = [
        SpeechSpeakerSegment(idx=0, start=0.0, end=1.0, speaker="SPEAKER_00", duration=1.0, prob=1.0),
        SpeechSpeakerSegment(idx=1, start=2.0, end=3.0, speaker="SPEAKER_01", duration=1.0, prob=1.0),
    ]
    assert result == expected