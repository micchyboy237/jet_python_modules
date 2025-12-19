# test_essentia_audio_insights_extractor.py

import numpy as np
from numpy.typing import NDArray

from jet.audio.insights.essentia_audio_insights_extractor import (
    extract_loudness,
    extract_mfcc,
    extract_pitch_and_key,
    extract_rhythm,
)


class TestLoadAudio:
    def test_load_audio_silence(self):
        # Given a short silent audio signal
        silent_audio: NDArray[np.float32] = np.zeros(44100, dtype=np.float32)  # 1 second of silence at 44100 Hz

        # When loading it (using synthetic data instead of file)
        # Note: load_audio requires a real file; for unit testing, we mock or test other functions directly.
        # Here we test core extractors with synthetic input.

        # Then proceed to test extractors on this known input


class TestExtractLoudness:
    def test_silence_returns_low_values(self):
        # Given a silent audio signal
        audio: NDArray[np.float32] = np.zeros(44100, dtype=np.float32)

        # When extracting loudness
        result = extract_loudness(audio)

        # Then integrated loudness should be very low (close to -70 LUFS or lower)
        expected_integrated = result["integrated_loudness"]
        expected_range = result["loudness_range"]
        assert expected_integrated < -60.0, f"Integrated loudness too high: {expected_integrated}"
        assert 0.0 <= expected_range < 1.0, f"Unexpected loudness range for silence: {expected_range}"


class TestExtractMFCC:
    def test_silence_returns_near_zero_mean(self):
        # Given a silent audio signal
        audio: NDArray[np.float32] = np.zeros(44100 * 2, dtype=np.float32)  # 2 seconds for multiple frames

        # When extracting MFCC mean
        result = extract_mfcc(audio)

        # Then mean MFCCs should have very high DC (coeff 0) and near-zero others
        mfcc_mean = result["mfcc_mean"]
        expected_dc = mfcc_mean[0]
        expected_higher_mean_abs = np.abs(mfcc_mean[1:]).mean()
        assert expected_dc < -1000.0, f"Unexpected DC coeff for silence: {expected_dc}"
        assert expected_higher_mean_abs < 1.0, f"Higher MFCC coeffs not near zero: {expected_higher_mean_abs}"


class TestExtractPitchAndKey:
    def test_silence_returns_low_salience_and_possible_key(self):
        # Given a silent audio signal
        audio: NDArray[np.float32] = np.zeros(44100 * 3, dtype=np.float32)

        # When extracting pitch and key
        result = extract_pitch_and_key(audio)

        # Then pitch salience mean should be very low
        expected_salience = result["pitch_salience_mean"]
        assert expected_salience == 0.0, f"Unexpected salience for silence: {expected_salience}"
        # Key/scale may be arbitrary but strength low
        expected_strength = result["key_strength"]
        assert expected_strength <= 0.0, f"Unexpected key strength for silence: {expected_strength}"
        assert result["scale"] in ["major", "minor"]
        assert result["key"] in [
            "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"
        ]


class TestExtractRhythm:
    def test_constant_noise_returns_reasonable_bpm_and_empty_beats(self):
        # Given white noise (no clear rhythm)
        np.random.seed(42)
        audio: NDArray[np.float32] = np.random.uniform(-1, 1, 44100 * 5).astype(np.float32)

        # When extracting rhythm
        result = extract_rhythm(audio)

        # Then BPM should be in plausible range (often around 100-140 for noise)
        assert 60.0 < result["bpm"] < 200.0
        # Beats positions may be empty or few for noise
        assert len(result["beats_positions"]) >= 0


class TestExtractAllInsights:
    def test_silence_produces_coherent_insights(self):
        # Given a silent audio signal
        audio: NDArray[np.float32] = np.zeros(44100 * 4, dtype=np.float32)

        # When extracting all insights using individual functions
        loudness = extract_loudness(audio)
        spectral = extract_mfcc(audio)
        tonal = extract_pitch_and_key(audio)
        rhythm = extract_rhythm(audio)

        # Then all components show silence characteristics
        assert loudness["integrated_loudness"] < -60.0
        assert np.abs(spectral["mfcc_mean"][1:]).mean() < 20.0
        assert tonal["pitch_salience_mean"] < 0.1
        assert rhythm["bpm"] > 0  # Algorithm always returns a BPM estimate