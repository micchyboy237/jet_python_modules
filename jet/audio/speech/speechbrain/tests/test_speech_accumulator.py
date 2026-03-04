"""Unit tests for LiveSpeechSegmentAccumulator."""

from collections import deque

import pytest
from jet.audio.speech.speechbrain.speech_accumulator import (
    LiveSpeechSegmentAccumulator,
    SegmentStats,
)


@pytest.fixture
def empty_preroll() -> deque[bytes]:
    return deque(maxlen=10)


@pytest.fixture
def sample_preroll() -> deque[bytes]:
    d = deque(maxlen=5)
    # Simulate 3 × 512-sample chunks of silence-ish data
    silence_chunk = bytes([0x00, 0x00] * 512)
    for _ in range(3):
        d.append(silence_chunk)
    return d


@pytest.fixture
def accumulator_no_preroll(empty_preroll: deque[bytes]) -> LiveSpeechSegmentAccumulator:
    return LiveSpeechSegmentAccumulator(
        sample_rate=16000, pre_roll_buffer=empty_preroll
    )


def make_chunk() -> bytes:
    """Helper: 512 samples × 2 bytes = 1024 bytes chunk"""
    return bytes([0x00, 0x40] * 512)  # arbitrary non-zero data


def test_creation_empty_no_preroll(accumulator_no_preroll):
    """Given a fresh accumulator with no pre-roll
    When we just created it
    Then buffer is empty and stats are zeroed safely
    """
    acc = accumulator_no_preroll

    assert len(acc.buffer) == 0
    assert acc.speech_chunk_count == 0
    assert acc.vad_sum == 0.0
    assert acc.vad_min == 1.0
    assert acc.vad_max == 0.0
    assert acc.energy_min == float("inf")
    assert acc.energy_max == 0.0

    stats: SegmentStats = acc.get_stats()
    assert stats["speech_chunk_count"] == 0
    assert stats["vad_sum"] == 0.0
    assert stats["vad_min"] == 0.0  # safe default
    assert stats["vad_max"] == 0.0
    assert stats["energy_min"] == 0.0
    assert stats["energy_max"] == 0.0
    assert stats["duration_ms"] == pytest.approx(0.0)
    assert acc.get_duration_sec() == pytest.approx(0.0)


def test_creation_with_preroll(sample_preroll):
    """Given some pre-roll chunks
    When accumulator is created
    Then pre-roll is already in buffer (but not counted as speech chunks)
    """
    acc = LiveSpeechSegmentAccumulator(
        sample_rate=16000, pre_roll_buffer=sample_preroll
    )

    # 3 chunks × 1024 bytes
    expected_preroll_bytes = 3 * 1024
    assert len(acc.buffer) == expected_preroll_bytes
    assert acc.speech_chunk_count == 0  # pre-roll doesn't count as speech
    expected_sec = 3 * 512 / 16000
    assert acc.get_duration_sec() == pytest.approx(expected_sec, abs=1e-6)
    assert acc.get_stats()["duration_ms"] == pytest.approx(
        expected_sec * 1000, abs=1e-6
    )
    assert acc.get_end_wallclock() == pytest.approx(
        acc.start_time + expected_sec, abs=1e-6
    )


def test_append_one_chunk(accumulator_no_preroll):
    """Given fresh accumulator
    When we append one chunk with known prob and rms
    Then stats reflect exactly that one value
    """
    acc = accumulator_no_preroll

    chunk = make_chunk()
    vad_prob = 0.72
    rms = 0.184

    acc.append(chunk, vad_prob, rms)

    assert acc.speech_chunk_count == 1
    assert len(acc.buffer) == 1024

    stats = acc.get_stats()
    assert stats["speech_chunk_count"] == 1
    assert stats["vad_sum"] == pytest.approx(0.72)
    assert stats["vad_min"] == pytest.approx(0.72)
    assert stats["vad_max"] == pytest.approx(0.72)
    assert stats["energy_sum"] == pytest.approx(0.184)
    assert stats["energy_min"] == pytest.approx(0.184)
    assert stats["energy_max"] == pytest.approx(0.184)

    expected_sec = 512 / 16000
    assert acc.get_duration_sec() == pytest.approx(expected_sec, abs=1e-6)
    assert stats["duration_ms"] == pytest.approx(expected_sec * 1000, abs=1e-6)


def test_multiple_appends_accumulation(accumulator_no_preroll):
    """Given fresh accumulator
    When we append several chunks with different values
    Then min/max/avg are correct
    """
    acc = accumulator_no_preroll

    values = [
        (0.65, 0.12),
        (0.92, 0.31),
        (0.41, 0.08),
        (0.88, 0.27),
    ]

    for prob, rms in values:
        acc.append(make_chunk(), prob, rms)

    stats = acc.get_stats()
    n = len(values)

    assert stats["speech_chunk_count"] == n
    assert stats["vad_min"] == pytest.approx(min(p for p, _ in values))
    assert stats["vad_max"] == pytest.approx(max(p for p, _ in values))
    assert stats["vad_sum"] == pytest.approx(sum(p for p, _ in values))

    rms_values = [r for _, r in values]
    assert stats["energy_min"] == pytest.approx(min(rms_values))
    assert stats["energy_max"] == pytest.approx(max(rms_values))
    assert stats["energy_sum"] == pytest.approx(sum(rms_values))

    expected_avg_vad = sum(p for p, _ in values) / n
    expected_avg_rms = sum(r for _, r in values) / n

    # rough check on std (we don't store it directly, but can recompute)
    variance = (stats["energy_sum_squares"] / n) - (expected_avg_rms**2)
    assert variance >= 0.0

    expected_sec = n * 512 / 16000
    assert acc.get_duration_sec() == pytest.approx(expected_sec, abs=1e-6)
    assert stats["duration_ms"] == pytest.approx(expected_sec * 1000, abs=1e-6)
    assert acc.get_end_wallclock() == pytest.approx(
        acc.start_time + expected_sec, abs=1e-6
    )


def test_reset_brings_back_to_clean_state(accumulator_no_preroll):
    """Given accumulator with data
    When we call reset()
    Then stats are cleared, buffer empty, but class still usable
    """
    acc = accumulator_no_preroll

    for i in range(4):
        acc.append(make_chunk(), 0.6 + i * 0.05, 0.15 + i * 0.03)

    assert acc.speech_chunk_count == 4
    assert len(acc.buffer) > 0

    acc.reset()

    assert acc.speech_chunk_count == 0
    assert len(acc.buffer) == 0
    assert acc.vad_min == 1.0
    assert acc.vad_max == 0.0
    assert acc.energy_min == float("inf")
    assert acc.energy_max == 0.0

    stats = acc.get_stats()
    assert stats["speech_chunk_count"] == 0
    assert stats["vad_min"] == 0.0
    assert stats["vad_max"] == 0.0
    assert stats["energy_min"] == 0.0
    assert stats["duration_ms"] == 0.0
    assert acc.get_duration_sec() == 0.0
    assert acc.get_end_wallclock() == pytest.approx(acc.start_time, abs=1e-6)


def test_duration_samples():
    acc = LiveSpeechSegmentAccumulator(16000, deque())

    acc.append(make_chunk(), 0.7, 0.2)
    acc.append(make_chunk(), 0.7, 0.2)

    assert acc.duration_samples() == 512 * 2

    expected_sec = 1024 / 16000
    assert acc.get_duration_sec() == pytest.approx(expected_sec, abs=1e-6)
    assert acc.get_stats()["duration_ms"] == pytest.approx(
        expected_sec * 1000, abs=1e-6
    )


def test_has_data_flag(accumulator_no_preroll):
    """Verify 'has_data' behaves correctly across lifecycle states."""
    acc = accumulator_no_preroll

    # ─── Given: freshly created accumulator ───────────────────────────────
    stats = acc.get_stats()

    # ─── Then: no data yet ────────────────────────────────────────────────
    assert stats["has_data"] is False
    assert stats["speech_chunk_count"] == 0
    assert stats["vad_min"] == 0.0
    assert stats["vad_max"] == 0.0
    assert stats["energy_min"] == 0.0
    assert stats["energy_max"] == 0.0


def test_has_data_after_append(accumulator_no_preroll):
    """After appending chunks, has_data becomes True."""
    acc = accumulator_no_preroll

    # ─── When: append one chunk ───────────────────────────────────────────
    acc.append(make_chunk(), speech_prob=0.68, rms=0.19)

    stats = acc.get_stats()

    # ─── Then: has meaningful data ────────────────────────────────────────
    assert stats["has_data"] is True
    assert stats["speech_chunk_count"] == 1
    assert stats["vad_min"] == pytest.approx(0.68)
    assert stats["vad_max"] == pytest.approx(0.68)
    assert stats["energy_min"] == pytest.approx(0.19)
    assert stats["energy_max"] == pytest.approx(0.19)

    # ─── When: append second chunk ────────────────────────────────────────
    acc.append(make_chunk(), speech_prob=0.92, rms=0.33)

    stats = acc.get_stats()

    assert stats["has_data"] is True
    assert stats["speech_chunk_count"] == 2
    assert stats["vad_min"] == pytest.approx(0.68)  # min of the two
    assert stats["vad_max"] == pytest.approx(0.92)


def test_has_data_after_reset(accumulator_no_preroll):
    """After reset, has_data returns to False even if data existed before."""
    acc = accumulator_no_preroll

    # First fill it
    for i in range(3):
        acc.append(make_chunk(), 0.5 + i * 0.1, 0.1 + i * 0.05)

    assert acc.get_stats()["has_data"] is True
    assert acc.get_stats()["speech_chunk_count"] == 3

    # ─── When: reset ──────────────────────────────────────────────────────
    acc.reset()

    stats = acc.get_stats()

    # ─── Then: back to no-data state ──────────────────────────────────────
    assert stats["has_data"] is False
    assert stats["speech_chunk_count"] == 0
    assert stats["vad_min"] == 0.0
    assert stats["vad_max"] == 0.0
    assert stats["energy_min"] == 0.0
    assert stats["energy_max"] == 0.0
    assert len(acc.buffer) == 0


def test_trim_audio_keeps_most_recent():
    acc = LiveSpeechSegmentAccumulator(16000, deque())

    print("Chunk size:", len(make_chunk()))  # ← MUST be 1024

    for i in range(20):
        prob = 0.3 if i < 8 else 0.85
        rms = 0.08 + i * 0.015
        acc.append(make_chunk(), prob, rms)

    print("Final buffer size:", len(acc.buffer))  # ← should be ~20480
    print("Reported duration:", acc.get_duration_sec())

    original_dur = acc.get_duration_sec()
    assert 0.63 < original_dur < 0.65

    acc.trim_audio(0.4)

    print("After trim - buffer size:", len(acc.buffer))
    print("After trim - duration:", acc.get_duration_sec())

    new_dur = acc.get_duration_sec()
    assert 0.38 < new_dur <= 0.42

    stats = acc.get_stats()
    # With simple tail trim → should keep high-confidence part
    assert stats["vad_min"] >= 0.84  # close to 0.85


def test_trim_audio_very_short_target():
    acc = LiveSpeechSegmentAccumulator(16000, deque())
    for _ in range(10):
        acc.append(make_chunk(), 0.9, 0.3)

    acc.trim_audio(0.7)

    dur = acc.get_duration_sec()
    assert dur <= 0.8 + 1e-6  # allow float imprecision
    assert len(acc._vad_probs) > 0  # at least 1 chunk kept


def test_trim_audio_zero_or_negative():
    acc = LiveSpeechSegmentAccumulator(16000, deque())
    acc.append(make_chunk(), 0.8, 0.25)
    assert len(acc.buffer) > 0

    acc.trim_audio(0)
    assert len(acc.buffer) == 0
    assert acc.get_duration_sec() == 0.0

    acc.append(make_chunk(), 0.7, 0.2)
    acc.trim_audio(-1)
    assert len(acc.buffer) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Pre-roll specific tests (new)
# ─────────────────────────────────────────────────────────────────────────────


def test_preroll_respects_max_duration_cap():
    """Given a pre-roll buffer larger than allowed duration
    When max_pre_roll_duration_sec is set
    Then only trailing chunks within duration are used
    """
    sample_rate = 16000

    # 10 chunks total
    d = deque(maxlen=20)
    chunk = bytes([0x00, 0x00] * 512)
    for _ in range(10):
        d.append(chunk)

    # Each chunk = 512 / 16000 = 0.032 sec
    # 10 chunks ≈ 0.32 sec
    # Cap to ~0.096 sec (≈ 3 chunks)
    acc = LiveSpeechSegmentAccumulator(
        sample_rate=sample_rate,
        pre_roll_buffer=d,
        max_pre_roll_duration_sec=0.096,
    )

    expected_chunks = 3
    expected_bytes = expected_chunks * 1024
    expected_sec = expected_chunks * 512 / sample_rate

    assert len(acc.buffer) == expected_bytes
    assert acc.get_duration_sec() == pytest.approx(expected_sec, abs=1e-6)
    assert acc.speech_chunk_count == 0  # still no speech


def test_preroll_zero_duration_cap():
    """Given a pre-roll buffer
    When max_pre_roll_duration_sec is 0
    Then no pre-roll is used
    """
    d = deque(maxlen=10)
    chunk = bytes([0x00, 0x00] * 512)
    for _ in range(5):
        d.append(chunk)

    acc = LiveSpeechSegmentAccumulator(
        sample_rate=16000,
        pre_roll_buffer=d,
        max_pre_roll_duration_sec=0,
    )

    assert len(acc.buffer) == 0
    assert acc.get_duration_sec() == 0.0
    assert acc.speech_chunk_count == 0


def test_preroll_invalid_chunk_size_is_skipped():
    """Given a pre-roll buffer containing invalid chunk sizes
    When accumulator is created
    Then invalid chunks are ignored
    """
    d = deque(maxlen=10)

    valid_chunk = bytes([0x00, 0x00] * 512)
    invalid_chunk = b"\x00\x01\x02"  # wrong size

    d.append(valid_chunk)
    d.append(invalid_chunk)
    d.append(valid_chunk)

    acc = LiveSpeechSegmentAccumulator(
        sample_rate=16000,
        pre_roll_buffer=d,
    )

    # Only 2 valid chunks should remain
    assert len(acc.buffer) == 2 * 1024
    assert acc.get_duration_sec() == pytest.approx((2 * 512) / 16000, abs=1e-6)


def test_preroll_uses_trailing_chunks_only():
    """Given many pre-roll chunks
    When duration cap selects subset
    Then most recent chunks are kept (tail behavior)
    """
    sample_rate = 16000
    d = deque(maxlen=20)

    # Create distinguishable chunks
    for i in range(6):
        # first byte encodes index
        chunk = bytes([i, 0x00] * 512)
        d.append(chunk)

    # Each chunk ≈ 0.032 sec
    # Cap to 2 chunks
    acc = LiveSpeechSegmentAccumulator(
        sample_rate=sample_rate,
        pre_roll_buffer=d,
        max_pre_roll_duration_sec=0.064,
    )

    # Should keep last 2 chunks (i=4 and i=5)
    expected_first_byte_last_chunk = 4

    first_kept_byte = acc.buffer[0]
    assert first_kept_byte == expected_first_byte_last_chunk

    expected_sec = (2 * 512) / sample_rate
    assert acc.get_duration_sec() == pytest.approx(expected_sec, abs=1e-6)


def test_preroll_does_not_mutate_input_deque():
    """Given a pre-roll deque
    When accumulator is created
    Then original deque remains unchanged
    """
    d = deque(maxlen=10)
    chunk = bytes([0x00, 0x00] * 512)
    for _ in range(4):
        d.append(chunk)

    original_len = len(d)

    LiveSpeechSegmentAccumulator(
        sample_rate=16000,
        pre_roll_buffer=d,
        max_pre_roll_duration_sec=0.032,
    )

    assert len(d) == original_len  # no mutation
