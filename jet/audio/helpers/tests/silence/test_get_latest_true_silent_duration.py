"""
Unit tests for:
  - get_latest_true_silent_duration
  - has_true_silence_in_latest_frames

from jet.audio.helpers.silence
"""

from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers to build fake audio chunks
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
FRAME_LENGTH_SAMPLE = 400  # 25 ms @ 16 kHz
HOP_SIZE = 160  # 10 ms @ 16 kHz


def make_silent_chunk(n_samples: int = 1600) -> np.ndarray:
    """Return a chunk of zeros (guaranteed silent, energy == 0)."""
    return np.zeros(n_samples, dtype=np.float32)


def make_loud_chunk(n_samples: int = 1600, amplitude: float = 0.5) -> np.ndarray:
    """Return a chunk of non-silent audio (sine wave with high amplitude)."""
    t = np.linspace(0, 1, n_samples, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.float32)


# ---------------------------------------------------------------------------
# Module under test (imported lazily so patching works cleanly)
# ---------------------------------------------------------------------------


def _import():
    from jet.audio.helpers.silence import (
        get_latest_true_silent_duration,
        has_true_silence_in_latest_frames,
    )

    return get_latest_true_silent_duration, has_true_silence_in_latest_frames


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fns():
    return _import()


@pytest.fixture()
def get_dur(fns):
    return fns[0]


@pytest.fixture()
def has_sil(fns):
    return fns[1]


# ---------------------------------------------------------------------------
# Tests: get_latest_true_silent_duration
# ---------------------------------------------------------------------------


class TestGetLatestTrueSilentDuration:
    def test_all_silent_chunks(self, get_dur):
        """All chunks silent → duration equals total samples / sample_rate."""
        chunks = [make_silent_chunk(1600)] * 4  # 4 × 0.1 s = 0.4 s
        result = get_dur(chunks, threshold=0.01, sample_rate=SAMPLE_RATE)
        assert result == pytest.approx(0.4)

    def test_no_silent_chunks(self, get_dur):
        """No trailing silence → returns 0.0."""
        chunks = [make_loud_chunk()] * 4
        result = get_dur(chunks, threshold=0.01, sample_rate=SAMPLE_RATE)
        assert result == pytest.approx(0.0)

    def test_trailing_silence_only(self, get_dur):
        """Only the last two chunks are silent."""
        chunks = [
            make_loud_chunk(1600),
            make_loud_chunk(1600),
            make_silent_chunk(1600),
            make_silent_chunk(1600),
        ]
        result = get_dur(chunks, threshold=0.01, sample_rate=SAMPLE_RATE)
        # 2 silent chunks × 1600 samples / 16000 = 0.2 s
        assert result == pytest.approx(0.2)

    def test_silence_in_middle_not_counted(self, get_dur):
        """Silent chunk sandwiched between loud ones → 0.0 (no trailing silence)."""
        chunks = [
            make_loud_chunk(1600),
            make_silent_chunk(1600),
            make_loud_chunk(1600),
        ]
        result = get_dur(chunks, threshold=0.01, sample_rate=SAMPLE_RATE)
        assert result == pytest.approx(0.0)

    def test_leading_silence_not_counted(self, get_dur):
        """Silent chunks at the start followed by a loud chunk → 0.0."""
        chunks = [
            make_silent_chunk(1600),
            make_silent_chunk(1600),
            make_loud_chunk(1600),
        ]
        result = get_dur(chunks, threshold=0.01, sample_rate=SAMPLE_RATE)
        assert result == pytest.approx(0.0)

    def test_empty_list(self, get_dur):
        """Empty chunk list → 0.0."""
        result = get_dur([], threshold=0.01, sample_rate=SAMPLE_RATE)
        assert result == pytest.approx(0.0)

    def test_single_silent_chunk(self, get_dur):
        """Single silent chunk returns its own duration."""
        chunk = make_silent_chunk(3200)  # 0.2 s
        result = get_dur([chunk], threshold=0.01, sample_rate=SAMPLE_RATE)
        assert result == pytest.approx(0.2)

    def test_single_loud_chunk(self, get_dur):
        """Single loud chunk → 0.0."""
        result = get_dur(
            [make_loud_chunk(1600)], threshold=0.01, sample_rate=SAMPLE_RATE
        )
        assert result == pytest.approx(0.0)

    def test_variable_chunk_sizes(self, get_dur):
        """Handles chunks of different lengths correctly."""
        chunks = [
            make_loud_chunk(3200),
            make_silent_chunk(800),  # 0.05 s
            make_silent_chunk(1600),  # 0.10 s
        ]
        result = get_dur(chunks, threshold=0.01, sample_rate=SAMPLE_RATE)
        assert result == pytest.approx(0.15)

    def test_custom_sample_rate(self, get_dur):
        """Respects a non-default sample_rate."""
        chunks = [make_silent_chunk(8000)]  # 1 s at 8 kHz
        result = get_dur(chunks, threshold=0.01, sample_rate=8000)
        assert result == pytest.approx(1.0)

    def test_high_threshold_treats_loud_as_silent(self, get_dur):
        """A very high threshold causes loud chunks to be treated as silent too."""
        chunks = [make_loud_chunk(1600, amplitude=0.01)]  # low amplitude
        result = get_dur(chunks, threshold=10.0, sample_rate=SAMPLE_RATE)
        assert result == pytest.approx(0.1)

    def test_fully_silent_chunk_returns_true(self, has_sil):
        """All-silent chunk → True for any num_latest_frames."""
        chunk = make_silent_chunk(FRAME_LENGTH_SAMPLE * 10)
        assert has_sil(chunk, num_latest_frames=3, threshold=0.01) is True

    def test_fully_loud_chunk_returns_false(self, has_sil):
        """All-loud chunk → False."""
        chunk = make_loud_chunk(FRAME_LENGTH_SAMPLE * 10)
        assert has_sil(chunk, num_latest_frames=3, threshold=0.01) is False

    def test_silence_only_at_end(self, has_sil):
        """Non-silent leading frames + silent trailing frames → True."""
        loud_part = make_loud_chunk(FRAME_LENGTH_SAMPLE * 5)
        silent_part = make_silent_chunk(FRAME_LENGTH_SAMPLE * 5)
        chunk = np.concatenate([loud_part, silent_part])
        assert has_sil(chunk, num_latest_frames=3, threshold=0.01) is True

    def test_silence_only_at_start_not_detected_in_latest(self, has_sil):
        """Silent leading frames + loud trailing frames → False (silence not in latest)."""
        silent_part = make_silent_chunk(FRAME_LENGTH_SAMPLE * 5)
        loud_part = make_loud_chunk(FRAME_LENGTH_SAMPLE * 5)
        chunk = np.concatenate([silent_part, loud_part])
        assert has_sil(chunk, num_latest_frames=3, threshold=0.01) is False

    def test_num_latest_frames_one(self, has_sil):
        """num_latest_frames=1 inspects only the very last frame."""
        loud_part = make_loud_chunk(FRAME_LENGTH_SAMPLE * 4)
        silent_part = make_silent_chunk(FRAME_LENGTH_SAMPLE)
        chunk = np.concatenate([loud_part, silent_part])
        assert has_sil(chunk, num_latest_frames=1, threshold=0.01) is True

    def test_num_latest_frames_larger_than_total_frames(self, has_sil):
        """num_latest_frames > total frames → inspects all frames; returns True if any silent."""
        chunk = make_silent_chunk(FRAME_LENGTH_SAMPLE * 2)
        assert has_sil(chunk, num_latest_frames=100, threshold=0.01) is True

    def test_chunk_shorter_than_one_frame(self, has_sil):
        """Chunk shorter than frame_length is treated as a single frame."""
        short_silent = make_silent_chunk(FRAME_LENGTH_SAMPLE // 2)
        assert has_sil(short_silent, num_latest_frames=1, threshold=0.01) is True

        short_loud = make_loud_chunk(FRAME_LENGTH_SAMPLE // 2)
        assert has_sil(short_loud, num_latest_frames=1, threshold=0.01) is False

    def test_custom_frame_length_and_hop_size(self, has_sil):
        """Custom frame_length and hop_size are respected."""
        frame_len = 200
        hop = 100
        # Build: 5 loud frames, then 5 silent frames
        loud_part = make_loud_chunk(frame_len * 5)
        silent_part = make_silent_chunk(frame_len * 5)
        chunk = np.concatenate([loud_part, silent_part])
        assert (
            has_sil(
                chunk,
                num_latest_frames=3,
                threshold=0.01,
                frame_length=frame_len,
                hop_size=hop,
            )
            is True
        )

    def test_uses_calibrated_threshold_when_none(self, has_sil):
        """When threshold=None, calibrate_silence_threshold() is called."""
        chunk = make_silent_chunk(FRAME_LENGTH_SAMPLE * 4)
        with patch(
            "jet.audio.helpers.silence.calibrate_silence_threshold",
            return_value=0.01,
        ) as mock_cal:
            result = has_sil(chunk, num_latest_frames=2, threshold=None)
        mock_cal.assert_called_once()
        assert result is True

    def test_uses_calibrated_threshold_when_none_list(self, get_dur):
        """When threshold=None, calibrate_silence_threshold() is called."""
        chunks = [make_silent_chunk(1600)]
        with patch(
            "jet.audio.helpers.silence.calibrate_silence_threshold",
            return_value=0.01,
        ) as mock_cal:
            result = get_dur(chunks, threshold=None, sample_rate=SAMPLE_RATE)
        mock_cal.assert_called_once()
        assert result == pytest.approx(0.1)

    def test_mixed_frames_partial_silence_detected(self, has_sil):
        """Only one silent frame among several loud ones is enough to return True."""
        frames = [make_loud_chunk(FRAME_LENGTH_SAMPLE)] * 4
        frames.append(make_silent_chunk(FRAME_LENGTH_SAMPLE))  # last frame silent
        chunk = np.concatenate(frames)
        assert has_sil(chunk, num_latest_frames=2, threshold=0.01) is True

    def test_mixed_frames_silence_outside_window_not_detected(self, has_sil):
        """Silent frame exists but falls outside the latest-N window → False."""
        # Frame layout: [loud, loud, SILENT, loud, loud]
        frames = [make_loud_chunk(FRAME_LENGTH_SAMPLE)] * 2
        frames.append(make_silent_chunk(FRAME_LENGTH_SAMPLE))
        frames += [make_loud_chunk(FRAME_LENGTH_SAMPLE)] * 2
        chunk = np.concatenate(frames)
        # Only look at last 2 frames (both loud)
        assert has_sil(chunk, num_latest_frames=2, threshold=0.01) is False
