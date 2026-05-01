import unittest

from jet.audio.speech.vad_peak_analyzer import VADPeakAnalyzer, VADSegment


class TestVADActiveRegions(unittest.TestCase):
    """
    Tests for extract_active_regions() and the width-fix in extract_peaks/troughs.

    Plain-English mental model
    --------------------------
    Active region : a run of consecutive frames where speech probability
                    is AT or ABOVE the threshold — like a stretch of road
                    with the speed limit posted.
    Valley        : the opposite — frames BELOW a low threshold, i.e. silence.
    Width         : how wide (in frames) a peak or trough is at half its
                    height — previously always null because SciPy skipped the
                    calculation unless width was explicitly passed.
    """

    def setUp(self):
        self.analyzer = VADPeakAnalyzer(sample_rate=16000, frame_duration_ms=32.0)
        self.frame_s = 0.032  # seconds per frame

    # ------------------------------------------------------------------ #
    # WIDTH FIX                                                            #
    # ------------------------------------------------------------------ #

    def test_peak_width_is_not_null(self):
        """
        Peaks must now carry a numeric width instead of null.

        Before the fix, passing no `width` argument meant SciPy never
        computed widths, so details['width'] was always None.
        After the fix we pass width=0 internally so SciPy always runs
        the width calculation.
        """
        probs = [0.1, 0.2, 0.95, 0.3, 0.1]
        peaks = self.analyzer.extract_peaks(probs, height=0.5, prominence=0.1)
        self.assertEqual(len(peaks), 1)
        width_val = peaks[0]["details"]["width"]
        self.assertIsNotNone(width_val, "width should be a float, not None")
        self.assertIsInstance(width_val, float)
        self.assertGreater(width_val, 0.0, "a real peak must have positive width")

    def test_trough_width_is_not_null(self):
        """Troughs must also carry a numeric width after the fix."""
        probs = [0.9, 0.8, 0.15, 0.7, 0.85]
        troughs = self.analyzer.extract_troughs(probs, height=0.3, prominence=0.1)
        self.assertEqual(len(troughs), 1)
        width_val = troughs[0]["details"]["width"]
        self.assertIsNotNone(width_val, "trough width should be a float, not None")
        self.assertIsInstance(width_val, float)

    def test_explicit_width_filter_still_works(self):
        """
        Passing an explicit minimum width should still act as a filter.

        A very narrow spike (width ≈ 1 frame) should be dropped when we
        demand width >= 3, but kept when we ask for nothing.
        """
        # Narrow spike: one frame high, neighbours low
        probs = [0.1, 0.1, 0.95, 0.1, 0.1]
        peaks_no_filter = self.analyzer.extract_peaks(probs, height=0.5)
        peaks_wide_filter = self.analyzer.extract_peaks(probs, height=0.5, width=3)
        self.assertEqual(
            len(peaks_no_filter), 1, "spike should be found without width filter"
        )
        self.assertEqual(
            len(peaks_wide_filter), 0, "narrow spike should be rejected by width>=3"
        )

    # ------------------------------------------------------------------ #
    # ACTIVE REGIONS — basic behaviour                                     #
    # ------------------------------------------------------------------ #

    def test_empty_probs_returns_empty(self):
        """extract_active_regions on an empty list must return []."""
        result = self.analyzer.extract_active_regions([])
        self.assertEqual(result, [])

    def test_single_active_region(self):
        """
        A simple on-off signal should produce exactly one active region.

              frames:  0     1     2     3     4
              probs:  0.1   0.8   0.9   0.7   0.2
              active (>=0.5):  N     Y     Y     Y     N
                                └────── 1 region ──────┘
        """
        probs = [0.1, 0.8, 0.9, 0.7, 0.2]
        regions = self.analyzer.extract_active_regions(probs, threshold=0.5)
        self.assertEqual(len(regions), 1)

        r: VADSegment = regions[0]
        self.assertEqual(r["frame_start"], 1)
        self.assertEqual(r["frame_end"], 3)
        self.assertEqual(r["frame_length"], 3)
        self.assertAlmostEqual(r["start_s"], 1 * self.frame_s, places=4)
        self.assertAlmostEqual(r["end_s"], 4 * self.frame_s, places=4)  # end of frame 3
        self.assertAlmostEqual(r["duration_s"], 3 * self.frame_s, places=4)

    def test_multiple_active_regions(self):
        """
        Two separate bursts of speech should produce two regions.

              frames:  0    1    2    3    4    5    6
              probs:  0.9  0.8  0.1  0.05 0.85 0.9  0.1
              active:  Y    Y    N    N    Y    Y    N
                       └─region1─┘        └─region2─┘
        """
        probs = [0.9, 0.8, 0.1, 0.05, 0.85, 0.9, 0.1]
        regions = self.analyzer.extract_active_regions(probs, threshold=0.5)
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0]["frame_start"], 0)
        self.assertEqual(regions[0]["frame_end"], 1)
        self.assertEqual(regions[1]["frame_start"], 4)
        self.assertEqual(regions[1]["frame_end"], 5)

    def test_active_region_runs_to_end_of_signal(self):
        """
        A region that is still active at the last frame must be captured.
        This tests the 'handle open region at the end' branch.
        """
        probs = [0.1, 0.9, 0.95]
        regions = self.analyzer.extract_active_regions(probs, threshold=0.5)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["frame_end"], 2)  # last frame index

    def test_no_active_region_when_all_below_threshold(self):
        """All frames below threshold → no active regions."""
        probs = [0.1, 0.2, 0.3, 0.15]
        regions = self.analyzer.extract_active_regions(probs, threshold=0.5)
        self.assertEqual(regions, [])

    def test_all_frames_active(self):
        """All frames above threshold → one single region spanning everything."""
        probs = [0.7, 0.8, 0.9, 0.75]
        regions = self.analyzer.extract_active_regions(probs, threshold=0.5)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["frame_start"], 0)
        self.assertEqual(regions[0]["frame_end"], 3)
        self.assertEqual(regions[0]["frame_length"], 4)

    # ------------------------------------------------------------------ #
    # ACTIVE REGIONS — details dict                                        #
    # ------------------------------------------------------------------ #

    def test_active_region_details_keys(self):
        """The details dict must contain all expected keys."""
        probs = [0.1, 0.85, 0.9, 0.1]
        regions = self.analyzer.extract_active_regions(probs, threshold=0.5)
        details = regions[0]["details"]
        for key in (
            "threshold",
            "max_probability",
            "mean_probability",
            "frame_count",
            "region_probs",
        ):
            self.assertIn(key, details, f"Missing key: {key}")

    def test_active_region_details_values(self):
        """max/mean probability and frame_count must be numerically correct."""
        probs = [0.1, 0.6, 0.8, 0.1]
        regions = self.analyzer.extract_active_regions(probs, threshold=0.5)
        self.assertEqual(len(regions), 1)
        d = regions[0]["details"]
        self.assertAlmostEqual(d["max_probability"], 0.8, places=4)
        self.assertAlmostEqual(d["mean_probability"], 0.7, places=4)  # (0.6+0.8)/2
        self.assertEqual(d["frame_count"], 2)
        self.assertEqual(d["threshold"], 0.5)

    def test_active_region_custom_threshold(self):
        """Changing the threshold changes which frames count as active."""
        probs = [0.4, 0.6, 0.8, 0.4]
        # With threshold=0.5: frames 1,2 are active → 1 region
        regions_05 = self.analyzer.extract_active_regions(probs, threshold=0.5)
        # With threshold=0.7: only frame 2 is active → 1 region of length 1
        regions_07 = self.analyzer.extract_active_regions(probs, threshold=0.7)
        # With threshold=0.9: nothing active → 0 regions
        regions_09 = self.analyzer.extract_active_regions(probs, threshold=0.9)

        self.assertEqual(len(regions_05), 1)
        self.assertEqual(regions_05[0]["frame_length"], 2)

        self.assertEqual(len(regions_07), 1)
        self.assertEqual(regions_07[0]["frame_length"], 1)
        self.assertEqual(regions_07[0]["frame_start"], 2)

        self.assertEqual(len(regions_09), 0)

    # ------------------------------------------------------------------ #
    # INTEGRATION: peaks / troughs / active regions together              #
    # ------------------------------------------------------------------ #

    def test_peaks_fall_inside_active_regions(self):
        """
        Every detected peak should land inside at least one active region.
        This is an integration sanity-check: a peak by definition has high
        probability, so it must sit inside a speech-active stretch.
        """
        probs = [0.1, 0.2, 0.9, 0.85, 0.3, 0.1, 0.95, 0.4]
        peaks = self.analyzer.extract_peaks(probs, height=0.5, prominence=0.1)
        active_regions = self.analyzer.extract_active_regions(probs, threshold=0.5)

        active_frames = set()
        for r in active_regions:
            active_frames.update(range(r["frame_start"], r["frame_end"] + 1))

        for p in peaks:
            self.assertIn(
                p["frame_start"],
                active_frames,
                f"Peak at frame {p['frame_start']} is not inside any active region",
            )

    def test_troughs_fall_outside_active_regions(self):
        """
        Troughs (silence dips) should NOT fall inside active regions.
        """
        probs = [0.9, 0.85, 0.1, 0.05, 0.88, 0.92]
        troughs = self.analyzer.extract_troughs(probs, height=0.3, prominence=0.3)
        active_regions = self.analyzer.extract_active_regions(probs, threshold=0.5)

        active_frames = set()
        for r in active_regions:
            active_frames.update(range(r["frame_start"], r["frame_end"] + 1))

        for t in troughs:
            self.assertNotIn(
                t["frame_start"],
                active_frames,
                f"Trough at frame {t['frame_start']} incorrectly sits inside an active region",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
