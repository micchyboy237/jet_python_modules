# jet_python_modules/jet/audio/helpers/tests/test_energy.py
from pathlib import Path
import numpy as np
import pytest
from pytest import approx

from jet.audio.helpers.energy import (
    compute_energy,
    compute_energies,
    detect_sound,
    has_sound,
)

ASSETS_DIR = Path(__file__).parent / "assets"
SAMPLE_RATE = 16000


@pytest.fixture
def silent_audio() -> np.ndarray:
    """Pure silence: 1 second of zeros"""
    return np.zeros(SAMPLE_RATE, dtype=np.float32)


@pytest.fixture
def loud_beep() -> np.ndarray:
    """1-second 440Hz sine wave at -6 dBFS"""
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5


@pytest.fixture
def sample_speech_path(tmp_path: Path) -> Path:
    """Create a temporary WAV with speech-like content (beeps + silence)"""
    import soundfile as sf
    audio = np.concatenate([
        np.zeros(int(0.3 * SAMPLE_RATE)),           # 300ms silence
        np.sin(2 * np.pi * 800 * np.linspace(0, 0.1, int(0.1 * SAMPLE_RATE))) * 0.8,
        np.zeros(int(0.2 * SAMPLE_RATE)),
        np.sin(2 * np.pi * 1000 * np.linspace(0, 0.15, int(0.15 * SAMPLE_RATE))) * 0.9,
        np.zeros(int(0.25 * SAMPLE_RATE)),
    ]).astype(np.float32)
    wav_path = tmp_path / "speech_like.wav"
    sf.write(wav_path, audio, SAMPLE_RATE)
    return wav_path


@pytest.fixture
def pure_silence_path(tmp_path: Path) -> Path:
    import soundfile as sf
    audio = np.zeros(int(1.5 * SAMPLE_RATE), dtype=np.float32)
    wav_path = tmp_path / "pure_silence.wav"
    sf.write(wav_path, audio, SAMPLE_RATE)
    return wav_path


def test_compute_energy_silence(silent_audio):
    # Given a completely silent audio frame
    # When we compute energy
    result = compute_energy(silent_audio)
    # Then it should be exactly zero
    expected = 0.0
    assert result == expected


def test_compute_energy_loud_beep(loud_beep):
    result = compute_energy(loud_beep)
    # 2/π for amplitude 0.5, but with finite buffer → allow small error
    expected = 0.3183098861837907
    assert result == approx(expected, abs=1e-5)


def test_compute_energies_returns_correct_structure(sample_speech_path):
    results = compute_energies(sample_speech_path, chunk_duration=0.25)
    # synthetic file is ~1.0 s → exactly 4 chunks
    expected_chunk_count = 4
    assert len(results) == expected_chunk_count
    assert results[0]["start_s"] == 0.0
    assert results[0]["end_s"] == 0.25
    assert results[-1]["end_s"] == approx(1.0, abs=0.001)


def test_compute_energies_with_threshold_marks_silence(sample_speech_path):
    threshold = 0.01
    results = compute_energies(sample_speech_path, chunk_duration=0.25, silence_threshold=threshold)
    expected_is_silent = [True, False, False, True]   # matches our 4-chunk file
    assert [r["is_silent"] for r in results] == expected_is_silent


def test_detect_sound_returns_true_for_sound(loud_beep):
    # Given a loud chunk and reasonable threshold
    threshold = 0.1
    # When we call detect_sound
    result = detect_sound(loud_beep, threshold)
    # Then it detects sound
    expected = True
    assert result is expected


def test_detect_sound_returns_false_for_silence(silent_audio):
    # Given silence and threshold
    threshold = 0.001
    # When we call detect_sound
    result = detect_sound(silent_audio, threshold)
    # Then no sound
    expected = False
    assert result is expected


def test_has_sound_detects_speech_with_auto_threshold(sample_speech_path, monkeypatch):
    def mock_calibrate():
        return 0.05
    monkeypatch.setattr(
        "jet.audio.helpers.silence.calibrate_silence_threshold",  # correct module
        mock_calibrate
    )
    assert has_sound(sample_speech_path) is True


def test_has_sound_returns_false_for_pure_silence(pure_silence_path, monkeypatch):
    def mock_calibrate():
        return 0.01
    monkeypatch.setattr(
        "jet.audio.helpers.silence.calibrate_silence_threshold",
        mock_calibrate
    )
    assert has_sound(pure_silence_path) is False


def test_has_sound_respects_custom_threshold(sample_speech_path):
    # Given speech file and very high threshold
    result_high = has_sound(sample_speech_path, silence_threshold=0.9)
    # Then no sound detected
    expected_high = False
    assert result_high is expected_high

    # Given low threshold
    result_low = has_sound(sample_speech_path, silence_threshold=0.001)
    expected_low = True
    assert result_low is expected_low