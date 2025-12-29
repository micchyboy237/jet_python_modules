# tests/test_extract_audio_segment.py

from __future__ import annotations
from pathlib import Path
import io

import numpy as np
import pytest
import soundfile as sf

from jet.audio.utils import extract_audio_segment


@pytest.fixture
def temp_wav_file(tmp_path: Path):
    """
    Create a deterministic mono WAV file for testing.
    """
    sample_rate = 16000
    duration_sec = 2.0
    total_samples = int(sample_rate * duration_sec)
    audio = np.linspace(-1.0, 1.0, total_samples, dtype=np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(wav_path, audio, sample_rate, format="WAV", subtype="FLOAT")
    return wav_path, sample_rate, audio


@pytest.fixture
def wav_bytes(temp_wav_file):
    """
    Create WAV bytes from the fixture WAV file in float32 to avoid quantization.
    """
    wav_path, sample_rate, audio = temp_wav_file
    bio = io.BytesIO()
    sf.write(bio, audio, sample_rate, format="WAV", subtype="FLOAT")
    bio.seek(0)
    return bio.read(), sample_rate, audio


@pytest.fixture
def ndarray_audio():
    """
    Simple numpy array audio fixture.
    """
    sample_rate = 16000
    audio = np.linspace(-1.0, 1.0, 32000, dtype=np.float32)
    return audio, sample_rate


class TestExtractAudioSegment:
    # --- Tests for Path/str input ---
    def test_extract_full_audio_when_end_is_none(self, temp_wav_file):
        wav_path, sample_rate, original_audio = temp_wav_file
        result, sr = extract_audio_segment(wav_path, start=0.0, end=None)
        assert sr == sample_rate
        assert result.shape[0] == original_audio.shape[0]
        assert np.allclose(result, original_audio, rtol=1e-7, atol=1e-9)

    def test_extract_partial_audio_with_start_and_end(self, temp_wav_file):
        wav_path, sample_rate, original_audio = temp_wav_file
        start = 0.5
        end = 1.5
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        expected = original_audio[start_frame:end_frame]
        result, sr = extract_audio_segment(wav_path, start=start, end=end)
        assert sr == sample_rate
        assert result.shape[0] == expected.shape[0]
        assert np.allclose(result, expected, rtol=1e-7, atol=1e-9)

    def test_extract_from_start_until_end_of_file(self, temp_wav_file):
        wav_path, sample_rate, original_audio = temp_wav_file
        start = 1.0
        start_frame = int(start * sample_rate)
        expected = original_audio[start_frame:]
        result, sr = extract_audio_segment(wav_path, start=start, end=None)
        assert sr == sample_rate
        assert result.shape[0] == expected.shape[0]
        assert np.allclose(result, expected, rtol=1e-7, atol=1e-9)

    def test_raises_if_audio_file_does_not_exist(self, tmp_path: Path):
        missing_file = tmp_path / "missing.wav"
        with pytest.raises(FileNotFoundError):
            extract_audio_segment(missing_file, start=0.0, end=1.0)

    def test_raises_if_start_is_negative(self, temp_wav_file):
        wav_path, _, _ = temp_wav_file
        with pytest.raises(ValueError):
            extract_audio_segment(wav_path, start=-0.1, end=1.0)

    def test_raises_if_end_is_less_than_or_equal_to_start(self, temp_wav_file):
        wav_path, _, _ = temp_wav_file
        with pytest.raises(ValueError):
            extract_audio_segment(wav_path, start=1.0, end=1.0)

    def test_raises_if_start_beyond_audio_duration(self, temp_wav_file):
        wav_path, _, _ = temp_wav_file
        with pytest.raises(ValueError):
            extract_audio_segment(wav_path, start=10.0, end=None)

    # --- Tests for bytes input ---
    def test_extract_from_bytes(self, wav_bytes):
        data_bytes, sr, original_audio = wav_bytes
        start, end = 0.5, 1.0
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        expected = original_audio[start_frame:end_frame]

        result, sr_result = extract_audio_segment(data_bytes, start=start, end=end)
        assert sr_result == sr
        assert result.shape[0] == expected.shape[0]
        assert np.allclose(result, expected, rtol=1e-7, atol=1e-9)

    # --- Tests for ndarray input ---
    def test_extract_from_ndarray(self, ndarray_audio):
        audio, sr = ndarray_audio
        start, end = 0.25, 0.75
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        expected = audio[start_frame:end_frame]

        result, sr_result = extract_audio_segment(audio, start=start, end=end, sample_rate=sr)
        assert sr_result == sr
        assert result.shape[0] == expected.shape[0]
        assert np.allclose(result, expected, rtol=1e-7, atol=1e-9)

    def test_ndarray_missing_sample_rate_raises(self, ndarray_audio):
        audio, _ = ndarray_audio
        with pytest.raises(ValueError):
            extract_audio_segment(audio, start=0.0, end=0.5)

    # --- Test invalid type ---
    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            extract_audio_segment(12345, start=0.0, end=1.0)
