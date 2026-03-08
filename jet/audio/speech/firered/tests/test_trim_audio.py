import time

import pytest
from jet.audio.speech.firered.speech_accumulator import LiveSpeechSegmentAccumulator


# Helper to create a simple accumulator with controlled VAD/RMS values
def create_accumulator_with_data(
    sample_rate: int = 16000,
    num_chunks: int = 10,
    vad_values: list[float] | None = None,
    rms_values: list[float] | None = None,
    start_time: float | None = None,
) -> LiveSpeechSegmentAccumulator:
    acc = LiveSpeechSegmentAccumulator(
        sample_rate=sample_rate,
        start_time=start_time or time.monotonic(),
    )

    if vad_values is None:
        vad_values = [0.05] * num_chunks  # mostly silence by default
    if rms_values is None:
        rms_values = [0.02] * num_chunks

    chunk_bytes = b"\x00\x00" * 512  # 512 samples × 2 bytes = 1024 bytes of silence

    for i in range(num_chunks):
        vad = vad_values[i] if i < len(vad_values) else 0.05
        rms = rms_values[i] if i < len(rms_values) else 0.02
        acc.append(chunk_bytes, speech_prob=vad, rms=rms)

    return acc


class TestTrimAudio:
    def test_trim_audio_no_op_when_under_limit(self):
        acc = create_accumulator_with_data(num_chunks=8)

        original_duration = acc.get_duration_sec()
        original_start = acc.get_start_wallclock()
        original_chunk_count = acc.speech_chunk_count

        acc.trim_audio(max_duration=10.0)  # way above

        assert acc.get_duration_sec() == pytest.approx(original_duration, abs=0.001)
        assert acc.get_start_wallclock() == pytest.approx(original_start, abs=0.001)
        assert acc.speech_chunk_count == original_chunk_count
        assert len(acc._vad_probs) == original_chunk_count

    def test_trim_audio_truncates_from_front_when_exceeds(self):
        sample_rate = 16000
        chunk_duration = 512 / sample_rate  # ≈ 0.032 s

        acc = create_accumulator_with_data(
            sample_rate=sample_rate,
            num_chunks=40,
        )

        original_duration = acc.get_duration_sec()
        assert original_duration > 1.0

        max_duration = 0.5  # ≈ 15–16 chunks

        acc.trim_audio(max_duration)

        new_duration = acc.get_duration_sec()
        assert new_duration <= max_duration + 0.033  # allow one chunk tolerance
        assert new_duration > max_duration - 0.033

        # should have removed chunks from the beginning
        assert (
            acc.start_time > time.monotonic() - 10
        )  # rough check that start moved forward

    def test_trim_audio_respects_smart_trim_at_silence(self):
        sample_rate = 16000
        chunk_dur = 512 / sample_rate

        # Pattern: silence - low - high speech - low - silence - high - silence
        vad_pattern = [0.02, 0.04, 0.68, 0.71, 0.12, 0.65, 0.03, 0.02, 0.01]
        num_chunks = len(vad_pattern)

        acc = create_accumulator_with_data(
            sample_rate=sample_rate,
            num_chunks=num_chunks,
            vad_values=vad_pattern,
        )

        total_duration = acc.get_duration_sec()
        max_duration = total_duration - 3.5 * chunk_dur  # want to cut ~3–4 chunks

        acc.trim_audio(max_duration)

        # After trim we kept from index 2 onwards:
        # [0.68, 0.71, 0.12, 0.65, 0.03, 0.02, 0.01]
        # → contains both speech regions + trailing low VAD
        kept_vad = acc._vad_probs

        kept_chunks = len(kept_vad)
        max_allowed_chunks = int(max_duration / chunk_dur) + 1
        assert kept_chunks <= max_allowed_chunks + 2, (
            f"Smart trim kept {kept_chunks} chunks, but hard limit allows ~{max_allowed_chunks}"
        )

        assert len(kept_vad) == 7, (
            f"Expected to trim only the initial silence chunks → kept 7 instead of {len(kept_vad)}"
        )

        # Core invariant: smart trim should preserve high-confidence speech
        assert max(kept_vad) > 0.5, "should have kept at least one strong speech chunk"
        assert any(v > 0.5 for v in kept_vad), "should preserve clear speech regions"

        # Ends with low VAD (trailing silence / very low energy)
        assert kept_vad[-1] < 0.15, "should end at/near silence when possible"
        assert kept_vad[-2] < 0.15, "trailing portion should be low-energy"

    def test_smart_trim_finds_first_usable_silence_from_end(self):
        # VAD: high speech → silence → more speech → we should trim BEFORE the last silence
        vad_pattern = [0.72, 0.71, 0.68, 0.08, 0.04, 0.65, 0.62, 0.03]

        acc = create_accumulator_with_data(
            num_chunks=len(vad_pattern),
            vad_values=vad_pattern,
        )

        original_len = len(acc._vad_probs)
        acc.trim_audio(max_duration=0.15)  # force big trim (~4–5 chunks)

        kept_vad = acc._vad_probs
        assert len(kept_vad) >= 2, "should keep at least some trailing speech + silence"

        # We expect it kept part of the later speech burst + trailing silence
        assert any(v > 0.6 for v in kept_vad[:-1]), (
            "should have kept some high-confidence speech chunk"
        )
        assert kept_vad[-1] < 0.15, "should preferably end with (or near) silence"

    def test_trim_audio_handles_zero_max_duration(self):
        acc = create_accumulator_with_data(num_chunks=10)
        assert acc.speech_chunk_count > 0

        acc.trim_audio(0.0)
        assert acc.speech_chunk_count == 0
        assert len(acc.buffer) == 0
        assert len(acc._vad_probs) == 0
        assert acc.get_duration_sec() == 0.0

    def test_trim_audio_negative_or_invalid_max_duration_resets(self):
        acc = create_accumulator_with_data(num_chunks=6)

        original_count = acc.speech_chunk_count

        acc.trim_audio(-0.1)
        assert acc.speech_chunk_count == 0

        # refill
        acc = create_accumulator_with_data(num_chunks=6)
        acc.trim_audio(-999)
        assert acc.speech_chunk_count == 0

    def test_trim_does_not_break_stats_after_trimming(self):
        vad_pattern = [0.05, 0.07, 0.82, 0.79, 0.11, 0.03, 0.68, 0.71, 0.04]

        acc = create_accumulator_with_data(
            num_chunks=len(vad_pattern),
            vad_values=vad_pattern,
            rms_values=[0.01, 0.02, 0.45, 0.41, 0.03, 0.01, 0.38, 0.44, 0.02],
        )

        before = acc.get_stats()

        acc.trim_audio(max_duration=0.1)  # force trim

        after = acc.get_stats()

        assert after["speech_chunk_count"] > 0
        assert after["vad_min"] <= after["vad_max"]
        assert after["energy_min"] <= after["energy_max"]
        assert abs(after["duration_sec"] - len(acc._vad_probs) * (512 / 16000)) < 0.001

    def test_trim_on_empty_segment_is_noop(self):
        acc = LiveSpeechSegmentAccumulator(sample_rate=16000)
        assert acc.speech_chunk_count == 0

        acc.trim_audio(5.0)
        assert acc.speech_chunk_count == 0

        acc.trim_audio(0.0)
        assert acc.speech_chunk_count == 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
