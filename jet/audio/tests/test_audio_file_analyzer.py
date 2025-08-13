import pytest
import numpy as np
import soundfile as sf
from jet.audio.audio_file_analyzer import AudioFileAnalyzer
import os


@pytest.fixture
def temp_wav_file(tmp_path):
    """Create a temporary WAV file for testing."""
    file_path = tmp_path / "test_audio.wav"
    sample_rate = 44100
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    sf.write(file_path, audio_data, sample_rate, subtype='PCM_16')
    yield str(file_path)
    if os.path.exists(file_path):
        os.remove(file_path)


class TestAudioFileAnalyzer:
    """Test suite for AudioFileAnalyzer class."""

    def test_get_basic_metadata(self, temp_wav_file):
        """Test basic metadata extraction from a WAV file.

        Given: A valid WAV file with known properties
        When: get_basic_metadata is called
        Then: The metadata should match expected values
        """
        analyzer = AudioFileAnalyzer(temp_wav_file)
        result = analyzer.get_basic_metadata()
        expected = {
            "file_path": temp_wav_file,
            "file_format": "WAV",
            "sample_rate": 44100,
            "channels": 1,
            "duration_s": pytest.approx(1.0, abs=0.01),
            "file_size_bytes": pytest.approx(os.path.getsize(temp_wav_file), abs=100),
            "bit_depth": 16
        }
        for key, value in expected.items():
            assert key in result, f"Expected key {key} in metadata"
            if isinstance(value, float):
                assert result[key] == pytest.approx(
                    value, abs=0.01), f"Mismatch in {key}"
            else:
                assert result[key] == value, f"Mismatch in {key}"

    def test_get_audio_features(self, temp_wav_file):
        """Test audio feature extraction from a WAV file.

        Given: A WAV file with a 440 Hz sine wave
        When: get_audio_features is called
        Then: The features should be within expected ranges
        """
        analyzer = AudioFileAnalyzer(temp_wav_file)
        result = analyzer.get_audio_features()
        expected = {
            "mean_pitch_hz": pytest.approx(440.0, rel=0.1),
            "tempo_bpm": 0.0,  # Sine wave lacks rhythmic structure, expect no tempo
            "spectral_centroid_hz": pytest.approx(440.0, rel=0.1),
            # RMS for 0.5 amplitude sine wave
            "rms_energy": pytest.approx(0.35, rel=0.2)
        }
        for key, value in expected.items():
            assert key in result, f"Expected key {key} in features"
            assert result[key] == value, f"Mismatch in {key}: got {result[key]}, expected {value}"

    def test_analyze_invalid_file(self):
        """Test analysis with a non-existent file.

        Given: A non-existent file path
        When: analyze is called
        Then: Empty metadata and features should be returned
        """
        analyzer = AudioFileAnalyzer("non_existent_file.wav")
        result = analyzer.analyze()
        expected = {
            "metadata": {},
            "features": {}
        }
        assert result == expected, f"Expected empty result for invalid file, got {result}"
