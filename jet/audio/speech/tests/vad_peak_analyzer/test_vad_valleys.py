import unittest

from jet.audio.speech.vad_peak_analyzer import VADPeakAnalyzer, VADSegment


class TestVADValleys(unittest.TestCase):
    """
    Tests for extract_valleys() — contiguous silence/low-probability regions.

    Mental model
    ------------
    Valley  : a bowl-shaped stretch where speech probability stays BELOW a
              threshold.  The trough() method finds the single lowest point;
              extract_valleys() captures the whole silent bowl around it.

    Symmetry with active regions
    ----------------------------
    extract_active_regions(threshold=T)  →  frames where prob >= T   (speech)
    extract_valleys(threshold=T)         →  frames where prob <  T   (silence)
    Together they tile the probability axis: every frame is either speech or
    silence (with T being the dividing line).
    """

    def setUp(self):
        self.analyzer = VADPeakAnalyzer(sample_rate=16000, frame_duration_ms=32.0)
        self.frame_s = 0.032

    # ------------------------------------------------------------------ #
    # Basic behaviour                                                      #
    # ------------------------------------------------------------------ #

    def test_empty_probs_returns_empty(self):
        """extract_valleys on an empty list must return []."""
        self.assertEqual(self.analyzer.extract_valleys([]), [])

    def test_single_valley(self):
        """
        A simple high-low-high signal produces exactly one valley.

              frames:  0     1     2     3     4
              probs:  0.9   0.1   0.05  0.2   0.85
              silent (<0.3): N     Y     Y     Y     N
                              └──── 1 valley ──────┘
        """
        probs = [0.9, 0.1, 0.05, 0.2, 0.85]
        valleys = self.analyzer.extract_valleys(probs, threshold=0.3)
        self.assertEqual(len(valleys), 1)

        v: VADSegment = valleys[0]
        self.assertEqual(v["frame_start"], 1)
        self.assertEqual(v["frame_end"], 3)
        self.assertEqual(v["frame_length"], 3)
        self.assertAlmostEqual(v["start_s"], 1 * self.frame_s, places=4)
        self.assertAlmostEqual(v["end_s"], 4 * self.frame_s, places=4)
        self.assertAlmostEqual(v["duration_s"], 3 * self.frame_s, places=4)

    def test_multiple_valleys(self):
        """
        Two separate silence stretches produce two valley segments.

              frames:  0    1    2    3    4    5    6
              probs:  0.1  0.05 0.8  0.9  0.2  0.1  0.85
              silent:  Y    Y    N    N    Y    Y    N
                       └─valley1─┘        └─valley2─┘
        """
        probs = [0.1, 0.05, 0.8, 0.9, 0.2, 0.1, 0.85]
        valleys = self.analyzer.extract_valleys(probs, threshold=0.3)
        self.assertEqual(len(valleys), 2)
        self.assertEqual(valleys[0]["frame_start"], 0)
        self.assertEqual(valleys[0]["frame_end"], 1)
        self.assertEqual(valleys[1]["frame_start"], 4)
        self.assertEqual(valleys[1]["frame_end"], 5)

    def test_valley_runs_to_end_of_signal(self):
        """A valley open at the last frame must still be captured."""
        probs = [0.9, 0.1, 0.05]
        valleys = self.analyzer.extract_valleys(probs, threshold=0.3)
        self.assertEqual(len(valleys), 1)
        self.assertEqual(valleys[0]["frame_end"], 2)  # last frame index

    def test_no_valley_when_all_above_threshold(self):
        """All frames above threshold → no valleys."""
        probs = [0.7, 0.8, 0.9, 0.6]
        self.assertEqual(self.analyzer.extract_valleys(probs, threshold=0.3), [])

    def test_all_frames_silent(self):
        """All frames below threshold → one valley spanning everything."""
        probs = [0.1, 0.05, 0.2, 0.15]
        valleys = self.analyzer.extract_valleys(probs, threshold=0.3)
        self.assertEqual(len(valleys), 1)
        self.assertEqual(valleys[0]["frame_start"], 0)
        self.assertEqual(valleys[0]["frame_end"], 3)
        self.assertEqual(valleys[0]["frame_length"], 4)

    def test_frame_exactly_at_threshold_is_not_silent(self):
        """
        A frame whose probability equals the threshold is NOT in a valley
        (the condition is strictly less-than, not less-than-or-equal).
        """
        probs = [0.1, 0.3, 0.1]  # middle frame == threshold exactly
        valleys = self.analyzer.extract_valleys(probs, threshold=0.3)
        # frame 1 is AT threshold → not silent → valley is split into two
        # single-frame valleys (frames 0 and 2)
        self.assertEqual(len(valleys), 2)
        self.assertEqual(valleys[0]["frame_start"], 0)
        self.assertEqual(valleys[1]["frame_start"], 2)

    # ------------------------------------------------------------------ #
    # Details dict                                                         #
    # ------------------------------------------------------------------ #

    def test_valley_details_keys(self):
        """The details dict must contain all expected keys."""
        probs = [0.9, 0.1, 0.2, 0.9]
        valleys = self.analyzer.extract_valleys(probs, threshold=0.3)
        details = valleys[0]["details"]
        for key in (
            "threshold",
            "min_probability",
            "mean_probability",
            "frame_count",
            "region_probs",
        ):
            self.assertIn(key, details, f"Missing key: {key}")

    def test_valley_details_values(self):
        """min/mean probability and frame_count must be numerically correct."""
        probs = [0.9, 0.1, 0.2, 0.9]
        valleys = self.analyzer.extract_valleys(probs, threshold=0.3)
        self.assertEqual(len(valleys), 1)
        d = valleys[0]["details"]
        self.assertAlmostEqual(d["min_probability"], 0.1, places=4)
        self.assertAlmostEqual(d["mean_probability"], 0.15, places=4)  # (0.1+0.2)/2
        self.assertEqual(d["frame_count"], 2)
        self.assertEqual(d["threshold"], 0.3)

    def test_valley_custom_threshold(self):
        """Raising the threshold widens the valley; lowering it narrows it."""
        probs = [0.4, 0.6, 0.8, 0.4]
        # threshold=0.5: frames 0,3 are silent → 2 single-frame valleys
        v_05 = self.analyzer.extract_valleys(probs, threshold=0.5)
        # threshold=0.3: nothing is silent → 0 valleys
        v_03 = self.analyzer.extract_valleys(probs, threshold=0.3)

        self.assertEqual(len(v_05), 2)
        self.assertEqual(len(v_03), 0)

    # ------------------------------------------------------------------ #
    # Symmetry with active regions                                         #
    # ------------------------------------------------------------------ #

    def test_valleys_and_active_regions_are_complementary(self):
        """
        With the same threshold, every frame must belong to EITHER an active
        region OR a valley — never both, never neither.
        """
        probs = [0.1, 0.8, 0.9, 0.05, 0.7, 0.2, 0.85]
        T = 0.5
        active = self.analyzer.extract_active_regions(probs, threshold=T)
        valleys = self.analyzer.extract_valleys(probs, threshold=T)

        active_frames = set()
        for r in active:
            active_frames.update(range(r["frame_start"], r["frame_end"] + 1))

        valley_frames = set()
        for v in valleys:
            valley_frames.update(range(v["frame_start"], v["frame_end"] + 1))

        all_frames = set(range(len(probs)))

        # No frame appears in both sets
        self.assertEqual(
            active_frames & valley_frames,
            set(),
            "A frame cannot be both active and a valley",
        )

        # Together they cover all frames (no frame is missed)
        # Note: frames exactly AT threshold go to active (>=), not valley (<)
        self.assertEqual(
            active_frames | valley_frames,
            all_frames,
            "Every frame must be either active or a valley",
        )

    def test_troughs_fall_inside_valleys(self):
        """
        A single-frame trough (local minimum) must sit inside a valley region,
        because a trough by definition has low probability.
        """
        probs = [0.9, 0.85, 0.08, 0.05, 0.88, 0.92]
        troughs = self.analyzer.extract_troughs(probs, height=0.3, prominence=0.3)
        valleys = self.analyzer.extract_valleys(probs, threshold=0.3)

        valley_frames = set()
        for v in valleys:
            valley_frames.update(range(v["frame_start"], v["frame_end"] + 1))

        for t in troughs:
            self.assertIn(
                t["frame_start"],
                valley_frames,
                f"Trough at frame {t['frame_start']} should be inside a valley region",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
