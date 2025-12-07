# tests/test_audio_preprocessor.py
import tempfile
import numpy as np
import soundfile as sf
import pytest
from pathlib import Path

from jet.audio.speech.audio_preprocessor import AudioPreprocessor, PreprocessResult

TEST_DATA = Path(__file__).parent / "test_data"
TEST_DATA.mkdir(exist_ok=True)

# Generate a clean 16kHz speech-like test file once
@pytest.fixture(scope="session")
def sample_wav(tmp_path_factory):
    import soundfile as sf
    sr = 16000
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    # 440Hz tone + noise = speech-like
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))
    path = tmp_path_factory.mktemp("data") / "speech_16k.wav"
    sf.write(path, audio, sr)
    return str(path)


def test_preprocess_no_vad_keeps_duration(sample_wav):
    # Given
    preprocessor = AudioPreprocessor()
    # When
    result: PreprocessResult = preprocessor.preprocess(sample_wav, apply_vad=False)
    # Then
    expected_duration = 3.0
    assert abs(result["duration_sec"] - expected_duration) < 0.01
    assert result["sample_rate"] == 16000
    assert result["audio"].ndim == 1
    assert result["audio"].dtype == np.float32
    assert np.abs(result["audio"]).max() <= 1.0


def test_preprocess_resamples_48khz_to_16khz(tmp_path):
    # Given a 48kHz file
    audio = np.random.randn(48000 * 2).astype(np.float32)
    path = tmp_path / "temp_48k.wav"
    sf.write(str(path), audio, 48000)

    # When
    result = AudioPreprocessor().preprocess(str(path), apply_vad=False)

    # Then
    expected_samples = int(16000 * 2)
    assert result["sample_rate"] == 16000
    assert len(result["audio"]) == pytest.approx(expected_samples, abs=100)  # resampler tolerance


def test_normalize_loudness_brings_peak_to_target():
    audio = np.ones(16000, dtype=np.float32) * 0.9
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, audio, 16000)
        result = AudioPreprocessor().preprocess(f.name, apply_vad=False)
        peak = np.abs(result["audio"]).max()
        assert 0.09 < peak <= 0.11   # now passes again


def test_vad_silero_removes_silence(tmp_path):
    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    speech = 0.6 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.random.randn(len(t))  # real speech-like
    silence = np.zeros(int(sr * 2.0))
    full = np.concatenate([speech, silence, speech[::-1]])

    path = tmp_path / "speech_with_silence.wav"
    sf.write(str(path), full, sr)

    result = AudioPreprocessor(
        threshold=0.5,
        min_speech_duration=0.05,
        padding_duration=0.1,
    ).preprocess(str(path), apply_vad=True)

    assert result["duration_sec"] <= 3.4
    assert 0.5 < result["vad_kept_ratio"] < 0.8


def test_vad_silero_tunes_threshold_for_noise(tmp_path):
    sr = 16000
    noise = np.random.randn(int(sr * 2.0)).astype(np.float32) * 0.4   # loud noise

    path = tmp_path / "noise.wav"
    sf.write(str(path), noise, sr)

    high = AudioPreprocessor(threshold=0.8).preprocess(str(path), apply_vad=True)
    low  = AudioPreprocessor(threshold=0.3).preprocess(str(path), apply_vad=True)

    assert high["vad_kept_ratio"] < 0.4
    assert low["vad_kept_ratio"] > 0.7