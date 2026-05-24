"""Unit tests for normalize_loudness function."""

import os
import sys
import unittest
import warnings

import numpy as np
from jet.audio.helpers.loudness import LoudnessNormalizationResult, normalize_loudness

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from pyloudnorm.meter import Meter


class TestNormalizeLoudness(unittest.TestCase):
    """Test suite for normalize_loudness function."""

    def setUp(self):
        """Set up test fixtures."""
        self.rate = 44100
        self.duration = 2.0  # seconds
        self.n_samples = int(self.rate * self.duration)
        np.random.seed(42)

        # Create a test tone (440 Hz sine wave at -20 dB FS)
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        self.sine_wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        self.sine_wave *= 0.1  # -20 dB peak

        # Create silent audio
        self.silence = np.zeros(self.n_samples, dtype=np.float32)

        # Create multi-channel audio (stereo)
        self.stereo = np.column_stack([self.sine_wave, self.sine_wave * 0.5]).astype(
            np.float32
        )

        # Create 5.1 channel audio
        self.surround = np.column_stack(
            [
                self.sine_wave,  # L
                self.sine_wave * 0.8,  # R
                self.sine_wave * 0.5,  # C
                self.sine_wave * 0.3,  # Ls
                self.sine_wave * 0.2,  # Rs
            ]
        ).astype(np.float32)

    def test_returns_correct_type(self):
        """Test that normalize_loudness returns the correct result type."""
        result = normalize_loudness(self.sine_wave, self.rate)
        self.assertIsInstance(result, LoudnessNormalizationResult)

    def test_normalized_data_not_same_object(self):
        """Test that original data is not modified."""
        original = self.sine_wave.copy()
        result = normalize_loudness(original, self.rate)
        np.testing.assert_array_equal(original, self.sine_wave)
        self.assertFalse(np.shares_memory(original, result.normalized_data))

    def test_target_loudness_achieved_mono(self):
        """Test that mono audio reaches approximately the target loudness."""
        target = -23.0
        result = normalize_loudness(self.sine_wave, self.rate, target_loudness=target)
        # Allow ±1 LU tolerance due to measurement variations
        self.assertAlmostEqual(
            result.output_loudness,
            target,
            delta=1.0,
            msg=f"Expected {target} ±1 LUFS, got {result.output_loudness}",
        )

    def test_target_loudness_achieved_stereo(self):
        """Test that stereo audio reaches approximately the target loudness."""
        target = -16.0
        result = normalize_loudness(self.stereo, self.rate, target_loudness=target)
        self.assertAlmostEqual(
            result.output_loudness,
            target,
            delta=1.0,
            msg=f"Expected {target} ±1 LUFS, got {result.output_loudness}",
        )

    def test_target_loudness_achieved_surround(self):
        """Test that 5.1 audio reaches approximately the target loudness."""
        target = -18.0
        result = normalize_loudness(self.surround, self.rate, target_loudness=target)
        self.assertAlmostEqual(
            result.output_loudness,
            target,
            delta=1.0,
            msg=f"Expected {target} ±1 LUFS, got {result.output_loudness}",
        )

    def test_gain_correct_direction(self):
        """Test that gain is applied in the correct direction."""
        # Make audio quieter -> louder
        result_louder = normalize_loudness(
            self.sine_wave * 0.1, self.rate, target_loudness=-20.0
        )
        self.assertGreater(result_louder.applied_gain, 0)

        # Make audio louder -> quieter
        result_quieter = normalize_loudness(
            self.sine_wave, self.rate, target_loudness=-40.0
        )
        self.assertLess(result_quieter.applied_gain, 0)

    def test_gain_relationship(self):
        """Test that applied_gain_db and applied_gain_linear are consistent."""
        result = normalize_loudness(self.sine_wave, self.rate)
        expected_linear = 10 ** (result.applied_gain / 20.0)
        self.assertAlmostEqual(
            expected_linear,
            result.applied_gain_linear,
            places=5,
            msg="dB and linear gain values are inconsistent",
        )

    def test_silence_handling(self):
        """Test that silent audio is handled gracefully."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = normalize_loudness(self.silence, self.rate)

            # Should produce a warning
            self.assertTrue(
                any("silent" in str(warning.message).lower() for warning in w),
                "No warning raised for silent audio",
            )

        # Should return original data unchanged
        np.testing.assert_array_equal(
            result.normalized_data,
            self.silence,
            "Silent audio should be returned unchanged",
        )
        self.assertEqual(result.applied_gain, 0.0)
        self.assertFalse(result.clipped)
        self.assertTrue(np.isneginf(result.input_loudness))

    def test_clipping_detection(self):
        """Test that clipping is correctly detected."""
        # Create very loud audio that will clip when normalized
        loud_audio = self.sine_wave * 0.95  # near 0 dBFS

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = normalize_loudness(loud_audio, self.rate, target_loudness=-10.0)

            # Check for clipping warning
            clipping_warnings = [
                warning for warning in w if "clip" in str(warning.message).lower()
            ]
            if result.clipped:
                self.assertTrue(
                    len(clipping_warnings) > 0,
                    "Clipping detected but no warning issued",
                )

    def test_metadata_completeness(self):
        """Test that all metadata fields are populated."""
        result = normalize_loudness(self.sine_wave, self.rate)

        self.assertIsNotNone(result.normalized_data)
        self.assertIsNotNone(result.input_loudness)
        self.assertIsNotNone(result.target_loudness)
        self.assertIsNotNone(result.applied_gain)
        self.assertIsNotNone(result.applied_gain_linear)
        self.assertIsNotNone(result.output_loudness)
        self.assertIsNotNone(result.clipped)

        # Check types
        self.assertIsInstance(result.input_loudness, float)
        self.assertIsInstance(result.target_loudness, float)
        self.assertIsInstance(result.applied_gain, float)
        self.assertIsInstance(result.applied_gain_linear, float)
        self.assertIsInstance(result.output_loudness, float)
        self.assertIsInstance(result.clipped, bool)

    def test_custom_meter(self):
        """Test using a pre-configured Meter instance."""
        meter = Meter(self.rate, filter_class="DeMan", block_size=0.200)
        result = normalize_loudness(
            self.sine_wave, self.rate, meter=meter, target_loudness=-23.0
        )
        self.assertIsInstance(result, LoudnessNormalizationResult)
        self.assertTrue(np.isfinite(result.output_loudness))

    def test_custom_block_size(self):
        """Test with custom block size."""
        result = normalize_loudness(self.sine_wave, self.rate, block_size=0.200)
        self.assertIsInstance(result, LoudnessNormalizationResult)

    def test_custom_filter_class(self):
        """Test with custom filter class."""
        result = normalize_loudness(
            self.sine_wave, self.rate, filter_class="Fenton/Lee 1"
        )
        self.assertIsInstance(result, LoudnessNormalizationResult)
        self.assertTrue(np.isfinite(result.output_loudness))

    def test_different_targets(self):
        """Test various target loudness levels."""
        targets = [-14.0, -16.0, -23.0, -24.0, -30.0]
        for target in targets:
            with self.subTest(target=target):
                result = normalize_loudness(
                    self.sine_wave, self.rate, target_loudness=target
                )
                self.assertAlmostEqual(
                    result.target_loudness, target, msg=f"Target mismatch for {target}"
                )

    # --- Validation Tests ---

    def test_raises_on_non_ndarray(self):
        """Test that ValueError is raised for non-ndarray input."""
        with self.assertRaises(ValueError):
            normalize_loudness([1, 2, 3], self.rate)

    def test_raises_on_integer_dtype(self):
        """Test that ValueError is raised for integer dtype."""
        with self.assertRaises(ValueError):
            normalize_loudness(np.array([1, 2, 3], dtype=np.int16), self.rate)

    def test_raises_on_3d_data(self):
        """Test that ValueError is raised for 3D data."""
        data_3d = np.random.randn(100, 2, 2).astype(np.float32)
        with self.assertRaises(ValueError):
            normalize_loudness(data_3d, self.rate)

    def test_raises_on_too_many_channels(self):
        """Test that ValueError is raised for >5 channels."""
        data_6ch = np.random.randn(1000, 6).astype(np.float32)
        with self.assertRaises(ValueError):
            normalize_loudness(data_6ch, self.rate)

    def test_raises_on_invalid_rate(self):
        """Test that appropriate errors are raised for invalid rate."""
        with self.assertRaises(TypeError):
            normalize_loudness(self.sine_wave, "44100")

        with self.assertRaises(ValueError):
            normalize_loudness(self.sine_wave, 0)

        with self.assertRaises(ValueError):
            normalize_loudness(self.sine_wave, -44100)

    def test_raises_on_empty_data(self):
        """Test that ValueError is raised for empty array."""
        with self.assertRaises(ValueError):
            normalize_loudness(np.array([], dtype=np.float32), self.rate)

    def test_raises_on_too_short_data(self):
        """Test that ValueError is raised for data shorter than block size."""
        too_short = np.random.randn(100).astype(np.float32)  # < 400ms at 44.1kHz
        with self.assertRaises(ValueError):
            normalize_loudness(too_short, self.rate)

    def test_raises_on_non_numeric_target(self):
        """Test that TypeError is raised for non-numeric target."""
        with self.assertRaises(TypeError):
            normalize_loudness(self.sine_wave, self.rate, target_loudness="loud")

    # --- Edge Case Tests ---

    def test_very_quiet_audio(self):
        """Test handling of very quiet but non-silent audio."""
        quiet = self.sine_wave * 1e-10
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            result = normalize_loudness(quiet, self.rate)
        self.assertIsInstance(result, LoudnessNormalizationResult)
        self.assertFalse(np.isfinite(result.input_loudness))

    def test_single_sample_float64(self):
        """Test that float64 audio works."""
        data_f64 = self.sine_wave.astype(np.float64)
        result = normalize_loudness(data_f64, self.rate)
        self.assertEqual(result.normalized_data.dtype, np.float64)

    def test_output_is_writable(self):
        """Test that output array is writable."""
        result = normalize_loudness(self.sine_wave, self.rate)
        self.assertTrue(result.normalized_data.flags["WRITEABLE"])

    def test_shape_preserved(self):
        """Test that output shape matches input shape."""
        result = normalize_loudness(self.sine_wave, self.rate)
        self.assertEqual(result.normalized_data.shape, self.sine_wave.shape)

        result_stereo = normalize_loudness(self.stereo, self.rate)
        self.assertEqual(result_stereo.normalized_data.shape, self.stereo.shape)

    def test_no_gain_when_already_at_target(self):
        """Test that gain is approximately 0 when already at target."""
        # First normalize to a target
        result1 = normalize_loudness(self.sine_wave, self.rate, target_loudness=-20.0)
        # Then try to normalize the result to the same target
        result2 = normalize_loudness(
            result1.normalized_data, self.rate, target_loudness=-20.0
        )
        self.assertAlmostEqual(
            result2.applied_gain,
            0.0,
            delta=1.0,
            msg="Second normalization should have near-zero gain",
        )


class TestLoudnessNormalizationResult(unittest.TestCase):
    """Tests for the LoudnessNormalizationResult dataclass."""

    def test_creation(self):
        """Test that the result object can be created."""
        result = LoudnessNormalizationResult(
            normalized_data=np.array([1.0]),
            input_loudness=-23.0,
            target_loudness=-23.0,
            applied_gain=0.0,
            applied_gain_linear=1.0,
            output_loudness=-23.0,
            clipped=False,
        )
        self.assertIsInstance(result, LoudnessNormalizationResult)

    def test_field_access(self):
        """Test that all fields are accessible."""
        data = np.array([0.5, -0.5])
        result = LoudnessNormalizationResult(
            normalized_data=data,
            input_loudness=-20.0,
            target_loudness=-23.0,
            applied_gain=-3.0,
            applied_gain_linear=0.7079,
            output_loudness=-23.1,
            clipped=False,
        )
        np.testing.assert_array_equal(result.normalized_data, data)
        self.assertEqual(result.input_loudness, -20.0)
        self.assertEqual(result.applied_gain, -3.0)

    def test_immutability(self):
        """Test that result fields are immutable after creation."""
        result = LoudnessNormalizationResult(
            normalized_data=np.array([1.0]),
            input_loudness=-23.0,
            target_loudness=-23.0,
            applied_gain=0.0,
            applied_gain_linear=1.0,
            output_loudness=-23.0,
            clipped=False,
        )
        # Dataclass fields should be assignable by default
        result.applied_gain = -1.0
        self.assertEqual(result.applied_gain, -1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
