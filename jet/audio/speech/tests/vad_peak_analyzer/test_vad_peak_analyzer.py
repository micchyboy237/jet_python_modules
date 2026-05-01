import unittest

from jet.audio.speech.vad_peak_analyzer import VADPeakAnalyzer, VADSegment


class TestVADPeakAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up analyzer with common 32ms frames at 16kHz (typical for Silero VAD)."""
        self.analyzer = VADPeakAnalyzer(sample_rate=16000, frame_duration_ms=32.0)
        self.frame_duration_s = 0.032

    def test_empty_probs(self):
        """Test behavior with empty probability list."""
        peaks = self.analyzer.extract_peaks([])
        troughs = self.analyzer.extract_troughs([])
        self.assertEqual(peaks, [])
        self.assertEqual(troughs, [])

    def test_single_peak(self):
        """Test detection of a single clear peak."""
        probs = [0.1, 0.2, 0.95, 0.3, 0.1]

        peaks = self.analyzer.extract_peaks(probs, height=0.5, prominence=0.1)

        self.assertEqual(len(peaks), 1)
        p: VADSegment = peaks[0]

        self.assertEqual(p["frame_start"], 2)
        self.assertEqual(p["frame_end"], 2)
        self.assertEqual(p["frame_length"], 1)
        self.assertAlmostEqual(p["start_s"], 2 * self.frame_duration_s, places=4)
        self.assertAlmostEqual(p["end_s"], 3 * self.frame_duration_s, places=4)
        self.assertAlmostEqual(p["duration_s"], self.frame_duration_s, places=4)
        self.assertAlmostEqual(p["details"]["peak_probability"], 0.95, places=4)

    def test_multiple_peaks(self):
        """Test detection of multiple distinct peaks."""
        probs = [0.1, 0.8, 0.2, 0.1, 0.9, 0.3, 0.95, 0.4]

        peaks = self.analyzer.extract_peaks(
            probs, height=0.6, distance=2, prominence=0.1
        )

        self.assertEqual(len(peaks), 3)
        self.assertEqual([p["frame_start"] for p in peaks], [1, 4, 6])

    def test_single_trough(self):
        """Test detection of a single clear trough (minimum)."""
        probs = [0.9, 0.8, 0.15, 0.7, 0.85]

        troughs = self.analyzer.extract_troughs(probs, height=0.3, prominence=0.1)

        self.assertEqual(len(troughs), 1)
        t: VADSegment = troughs[0]

        self.assertEqual(t["frame_start"], 2)
        self.assertAlmostEqual(t["details"]["trough_probability"], 0.15, places=4)

    def test_multiple_troughs(self):
        """Test detection of multiple troughs."""
        probs = [0.8, 0.1, 0.7, 0.05, 0.9, 0.2, 0.85]

        troughs = self.analyzer.extract_troughs(
            probs, height=0.25, distance=1, prominence=0.05
        )

        self.assertEqual(len(troughs), 3)
        indices = [t["frame_start"] for t in troughs]
        self.assertEqual(indices, [1, 3, 5])

    def test_no_peaks_found(self):
        """Test when no peaks meet the criteria."""
        probs = [0.1, 0.2, 0.3, 0.25, 0.15]  # all low values

        peaks = self.analyzer.extract_peaks(probs, height=0.7, prominence=0.2)
        self.assertEqual(peaks, [])

    def test_no_troughs_found(self):
        """Test when no troughs meet the criteria."""
        probs = [0.9, 0.85, 0.95, 0.88]  # all high values

        troughs = self.analyzer.extract_troughs(probs, height=0.5)
        self.assertEqual(troughs, [])

    def test_peak_details_structure(self):
        """Test that details dictionary contains expected keys."""
        probs = [0.1, 0.05, 0.92, 0.3]

        peaks = self.analyzer.extract_peaks(probs, height=0.5, prominence=0.1)

        self.assertTrue(len(peaks) > 0)
        details = peaks[0]["details"]

        expected_keys = {"peak_index", "peak_probability", "prominence"}
        self.assertTrue(expected_keys.issubset(details.keys()))
        self.assertIsInstance(details["peak_probability"], float)

    def test_trough_details_structure(self):
        """Test trough details structure."""
        probs = [0.9, 0.12, 0.88]

        troughs = self.analyzer.extract_troughs(probs, height=0.3, prominence=0.05)

        self.assertTrue(len(troughs) > 0)
        details = troughs[0]["details"]

        self.assertIn("trough_index", details)
        self.assertIn("trough_probability", details)
        self.assertIsInstance(details["trough_probability"], float)

    def test_different_frame_duration(self):
        """Test analyzer with different frame duration (e.g. 10ms frames)."""
        analyzer10 = VADPeakAnalyzer(sample_rate=16000, frame_duration_ms=10.0)

        probs = [0.1, 0.2, 0.85, 0.3]

        peaks = analyzer10.extract_peaks(probs, height=0.5)

        self.assertEqual(len(peaks), 1)
        p = peaks[0]
        self.assertAlmostEqual(p["start_s"], 0.02, places=4)  # frame 2 * 0.01s
        self.assertAlmostEqual(p["duration_s"], 0.01, places=4)

    def test_prominence_filtering(self):
        """Test that prominence correctly filters weak peaks. Includes debug logging."""
        # Weak bump (prominence ≈ 0.45) vs strong isolated peak (prominence ≈ 0.83)
        probs = [0.10, 0.55, 0.52, 0.10, 0.95, 0.12]
        # Indices:     0     1     2     3     4     5

        print("\n=== Prominence Filtering Test ===")
        print(f"Input probs: {[round(p, 3) for p in probs]}")

        # Low prominence → should detect both peaks
        peaks_loose = self.analyzer.extract_peaks(
            probs, height=0.5, prominence=0.05, distance=2
        )
        print(
            f"Loose (prominence=0.05) → Found {len(peaks_loose)} peaks at {[p['frame_start'] for p in peaks_loose]}"
        )

        # High prominence → should keep ONLY the strong peak at index 4
        peaks_strict = self.analyzer.extract_peaks(
            probs, height=0.5, prominence=0.5, distance=2
        )
        print(
            f"Strict (prominence=0.5) → Found {len(peaks_strict)} peaks at {[p['frame_start'] for p in peaks_strict]}"
        )

        # Optional: show prominence values from the last call
        if peaks_strict:
            for p in peaks_strict:
                prom = p["details"].get("prominence")
                print(
                    f"  Peak at frame {p['frame_start']}: prob={p['details']['peak_probability']:.3f}, prominence={prom:.3f}"
                )

        self.assertEqual(
            len(peaks_loose),
            2,
            "Low prominence should detect both the weak bump and the strong peak",
        )

        self.assertEqual(
            len(peaks_strict),
            1,
            "prominence=0.5 should filter out the weak bump (prom≈0.45)",
        )
        self.assertEqual(
            peaks_strict[0]["frame_start"],
            4,
            "The surviving peak must be the strong one at index 4",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
