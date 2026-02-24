import numpy as np
import pytest
from jet.audio.audio_search import (
    find_audio_offset,
    find_audio_offsets,
    find_partial_audio_matches,
)


class TestAudioOffset:
    SAMPLE_RATE = 1000

    def test_exact_match_at_start(self):
        # Given a deterministic non-periodic signal
        long_signal = np.concatenate(
            [np.linspace(0, 1, 1000), np.random.default_rng(42).normal(0, 0.1, 2000)]
        )

        short_signal = long_signal[:500]

        # When searching for the short signal
        result = find_audio_offset(
            long_signal=long_signal,
            short_signal=short_signal,
            sample_rate=self.SAMPLE_RATE,
        )

        # Then it should match at index 0
        expected_start = 0
        expected_end = 500

        assert result is not None
        assert result["start_sample"] == expected_start
        assert result["end_sample"] == expected_end
        assert result["start_time"] == 0.0
        assert result["confidence"] == pytest.approx(1.0, abs=1e-6)

    def test_exact_match_with_offset(self):
        # Given silence + unique signal + silence
        rng = np.random.default_rng(123)

        silence = np.zeros(1000)
        unique = rng.normal(0, 1, 500)
        long_signal = np.concatenate([silence, unique, silence])

        short_signal = unique.copy()

        # When searching
        result = find_audio_offset(
            long_signal=long_signal,
            short_signal=short_signal,
            sample_rate=self.SAMPLE_RATE,
        )

        # Then it should detect correct offset
        expected_start = 1000
        expected_end = 1500

        assert result is not None
        assert result["start_sample"] == expected_start
        assert result["end_sample"] == expected_end
        assert result["start_time"] == 1.0
        assert result["end_time"] == 1.5
        assert result["confidence"] == pytest.approx(1.0, abs=1e-6)

    def test_returns_none_if_not_found(self):
        # Given two unrelated signals
        rng = np.random.default_rng(999)

        long_signal = rng.normal(0, 1, 3000)
        short_signal = rng.normal(0, 1, 500)

        # When searching with high confidence threshold
        result = find_audio_offset(
            long_signal=long_signal,
            short_signal=short_signal,
            sample_rate=self.SAMPLE_RATE,
            confidence_threshold=0.95,
        )

        # Then result should be None
        expected = None
        assert result is expected
        # INSERT_YOUR_CODE


class TestAudioOffsets:
    SAMPLE_RATE = 1000
    SHORT_LEN = 400

    def _make_long_with_two_matches(self):
        rng = np.random.default_rng(42)
        silence = np.zeros(800)
        part_a = np.sin(np.linspace(0, 20 * np.pi, self.SHORT_LEN)) + rng.normal(
            0, 0.08, self.SHORT_LEN
        )
        part_b = np.cos(np.linspace(0, 20 * np.pi, self.SHORT_LEN)) + rng.normal(
            0, 0.08, self.SHORT_LEN
        )
        long_signal = np.concatenate(
            [
                silence,
                part_a,
                silence * 2,
                part_a * 0.97,  # slightly lower amplitude → should still match well
                silence * 3,
                part_b,
                silence,
            ]
        )
        return long_signal, part_a, part_b

    def test_given_no_matches_when_searching_then_returns_empty_list(self):
        # Given
        rng = np.random.default_rng(777)
        long_signal = rng.normal(0, 1, 5000)
        short_signal = rng.normal(0, 1, 400)

        # When
        results = find_audio_offsets(
            long_signal=long_signal,
            short_signal=short_signal,
            sample_rate=self.SAMPLE_RATE,
            confidence_threshold=0.92,
        )

        # Then
        assert len(results) == 0

    def test_given_single_match_when_searching_then_returns_one_result(self):
        # Given
        long_signal = np.concatenate(
            [
                np.zeros(1200),
                np.linspace(0, 2, self.SHORT_LEN),
                np.zeros(1800),
            ]
        )
        short_signal = long_signal[1200 : 1200 + self.SHORT_LEN]

        # When
        results = find_audio_offsets(
            long_signal=long_signal,
            short_signal=short_signal,
            sample_rate=self.SAMPLE_RATE,
            confidence_threshold=0.98,  # we expect near-perfect match
        )

        # Then
        assert len(results) == 1
        match = results[0]
        assert match["start_sample"] == 1200
        assert match["end_sample"] == 1200 + self.SHORT_LEN
        assert match["confidence"] > 0.999

    def test_given_two_clear_matches_when_searching_then_finds_both(self):
        # Given
        long_signal, part_a, _ = self._make_long_with_two_matches()

        # When
        results = find_audio_offsets(
            long_signal=long_signal,
            short_signal=part_a,
            sample_rate=self.SAMPLE_RATE,
            confidence_threshold=0.985,  # very high → reject spurious side-lobes
            min_distance_samples=200,
        )

        # Then
        expected_starts = [800, 2000]
        found_starts = [r["start_sample"] for r in results]
        assert found_starts == expected_starts, (
            f"Expected starts {expected_starts}, got {found_starts}"
        )
        assert results[0]["start_sample"] < results[1]["start_sample"]
        assert results[0]["confidence"] > 0.97
        assert results[1]["confidence"] > 0.94  # slightly lower due to 0.97 amplitude

    def test_given_close_overlapping_matches_min_distance_suppresses_duplicates(self):
        # Given
        rng = np.random.default_rng(123)
        base = rng.normal(0, 1, self.SHORT_LEN)
        overlap_samples = 100
        long_signal = np.concatenate(
            [
                np.zeros(600),
                base,  # first copy starts at 600
                base[overlap_samples:]
                * 0.99,  # second copy starts overlap_samples earlier; overlap by 100 samples
                # → distance between starts = len(base) - overlap_samples = 300
                np.zeros(1500),
            ]
        )

        # When - tight min distance (allows close / overlapping matches)
        results_tight = find_audio_offsets(
            long_signal=long_signal,
            short_signal=base,
            sample_rate=self.SAMPLE_RATE,
            confidence_threshold=0.9,
            min_distance_samples=self.SHORT_LEN // 4,  # ~100
        )

        # When - wide min distance (should suppress near-duplicates / overlaps)
        results_wide = find_audio_offsets(
            long_signal=long_signal,
            short_signal=base,
            sample_rate=self.SAMPLE_RATE,
            confidence_threshold=0.9,
            min_distance_samples=int(
                self.SHORT_LEN * 0.9
            ),  # ~360 → should now suppress the overlapping duplicate
        )

        # Then
        assert len(results_tight) >= 1, (
            "Tight distance should find at least the best match"
        )
        assert len(results_wide) == 1, (
            "Wide distance (≈0.9×short_len) should suppress the overlapping near-duplicate"
        )
        # The best one should be kept in wide case
        if len(results_tight) >= 2:
            assert results_wide[0]["confidence"] >= results_tight[1]["confidence"], (
                "The kept match should have higher or equal confidence than the suppressed one"
            )


class TestPartialAudioMatches:
    SAMPLE_RATE = 1000
    SHORT_LEN = 600

    def _create_signals_with_partial(self, partial_fraction=0.7, position="middle"):
        rng = np.random.default_rng(42)
        silence = np.zeros(1200)
        core = np.sin(np.linspace(0, 24 * np.pi, self.SHORT_LEN)) + rng.normal(
            0, 0.06, self.SHORT_LEN
        )

        partial_len = int(self.SHORT_LEN * partial_fraction)

        if position == "prefix":
            short_part = core[:partial_len]
            long_part = core
        elif position == "suffix":
            short_part = core[-partial_len:]
            long_part = core
        else:  # middle
            short_part = core[
                int((self.SHORT_LEN - partial_len) / 2) : int(
                    (self.SHORT_LEN + partial_len) / 2
                )
            ]
            long_part = core

        long_signal = np.concatenate(
            [silence, long_part, silence * 2, core * 0.96, silence]
        )
        short_signal = short_part + rng.normal(0, 0.04, len(short_part))  # slight noise

        return long_signal, short_signal, partial_len

    def test_given_partial_prefix_match_when_searching_then_finds_it(self):
        # Given
        long_signal, short_signal, partial_len = self._create_signals_with_partial(
            0.65, "prefix"
        )

        # When
        results = find_partial_audio_matches(
            long_signal=long_signal,
            short_signal=short_signal,
            sample_rate=self.SAMPLE_RATE,
            confidence_threshold=0.78,
            min_match_fraction=0.5,
            max_match_fraction=1.0,
            length_step_fraction=0.15,
            min_distance_samples=200,
        )

        # Then
        expected = "Found at least one partial match"
        assert len(results) >= 1, expected

        best = results[0]
        assert best.match_length_samples >= int(len(short_signal) * 0.6)
        assert best.confidence > 0.80
        assert 1100 <= best.start_sample <= 1400, (
            "Should be near the beginning of the core signal"
        )

    def test_given_partial_suffix_match_then_finds_high_confidence(self):
        # Given
        long_signal, short_signal, partial_len = self._create_signals_with_partial(
            0.72, "suffix"
        )

        # When
        results = find_partial_audio_matches(
            long_signal=long_signal,
            short_signal=short_signal,
            sample_rate=self.SAMPLE_RATE,
            confidence_threshold=0.77,
            min_match_fraction=0.55,
        )

        # Then
        assert len(results) >= 1
        best = max(results, key=lambda x: x.confidence)
        assert best.confidence > 0.84
        assert best.match_length_samples >= int(len(short_signal) * 0.68)

    def test_given_middle_partial_then_recovers_correct_region(self):
        # Given
        long_signal, short_signal, partial_len = self._create_signals_with_partial(
            0.6, "middle"
        )

        # When
        results = find_partial_audio_matches(
            long_signal=long_signal,
            short_signal=short_signal,
            sample_rate=self.SAMPLE_RATE,
            confidence_threshold=0.75,
            min_match_fraction=0.45,
            length_step_fraction=0.12,
        )

        # Then
        assert len(results) > 0
        found_starts = [r.start_sample for r in results]
        assert any(1100 < s < 1500 for s in found_starts), (
            "Should find match in first core region"
        )

    def test_given_no_reasonable_partial_then_returns_empty(self):
        # Given
        rng = np.random.default_rng(123)
        long_signal = rng.normal(0, 1, 5000)
        short_signal = np.sin(np.linspace(0, 18 * np.pi, 700)) + rng.normal(0, 0.2, 700)

        # When
        results = find_partial_audio_matches(
            long_signal=long_signal,
            short_signal=short_signal,
            sample_rate=self.SAMPLE_RATE,
            confidence_threshold=0.82,
            min_match_fraction=0.6,
        )

        # Then
        assert len(results) == 0

    def test_given_full_match_is_also_found_as_partial(self):
        # Given
        rng = np.random.default_rng(99)
        silence = np.zeros(800)
        motif = rng.normal(0, 1, self.SHORT_LEN)
        long_signal = np.concatenate([silence, motif, silence])
        short_signal = motif.copy()

        # When
        results = find_partial_audio_matches(
            long_signal=long_signal,
            short_signal=short_signal + rng.normal(0, 0.03, len(short_signal)),
            sample_rate=self.SAMPLE_RATE,
            confidence_threshold=0.80,
            min_match_fraction=0.5,
        )

        # Then
        lengths = [r.match_length_samples for r in results]
        assert any(l >= self.SHORT_LEN * 0.95 for l in lengths), (
            "Full-length match should be found"
        )
        confidences = [r.confidence for r in results]
        assert max(confidences) > 0.96
