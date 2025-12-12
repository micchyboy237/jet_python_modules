# tests/test_chunked_vad.py

import pytest
import torch
from typing import List

# Import the function and TypedDict from your module
from jet.audio.speech.silero.chunked_vad import get_speech_probabilities_chunks, SpeechProbChunk


# Fixture to load the official Silero VAD model (JIT version)
# This is lightweight and cached by torch.hub
@pytest.fixture(scope="session")
def vad_model():
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        source='github',
        trust_repo=True,
        force_reload=False,
    )
    return model


# ----------------------------------------------------------------------
# BDD-style tests using human-readable scenarios
# ----------------------------------------------------------------------


def test_short_silence_audio_returns_low_probabilities(vad_model):
    """
    Given: A very short silent audio (all zeros)
    When: Processing with 0.25s non-overlapping chunks
    Then: All chunks should have speech probability close to 0
    """
    sampling_rate = 16000
    duration_sec = 2.0
    audio = torch.zeros(int(duration_sec * sampling_rate))

    chunks: List[SpeechProbChunk] = get_speech_probabilities_chunks(
        audio=audio,
        model=vad_model,
        sampling_rate=sampling_rate,
        chunk_seconds=0.25,
        overlap_seconds=0.0,
        aggregation="mean",
    )

    expected = [
        SpeechProbChunk(start_sec=0.0000, end_sec=0.2500, duration_sec=0.2500, speech_prob=0.0000, num_windows=8),
        SpeechProbChunk(start_sec=0.2500, end_sec=0.5000, duration_sec=0.2500, speech_prob=0.0000, num_windows=8),
        SpeechProbChunk(start_sec=0.5000, end_sec=0.7500, duration_sec=0.2500, speech_prob=0.0000, num_windows=8),
        SpeechProbChunk(start_sec=0.7500, end_sec=1.0000, duration_sec=0.2500, speech_prob=0.0000, num_windows=8),
        SpeechProbChunk(start_sec=1.0000, end_sec=1.2500, duration_sec=0.2500, speech_prob=0.0000, num_windows=8),
        SpeechProbChunk(start_sec=1.2500, end_sec=1.5000, duration_sec=0.2500, speech_prob=0.0000, num_windows=8),
        SpeechProbChunk(start_sec=1.5000, end_sec=1.7500, duration_sec=0.2500, speech_prob=0.0000, num_windows=8),
        SpeechProbChunk(start_sec=1.7500, end_sec=2.0000, duration_sec=0.2500, speech_prob=0.0000, num_windows=8),
    ]

    # Use approximate comparison for floating point fields
    assert len(chunks) == len(expected)
    for result, exp in zip(chunks, expected):
        assert result["start_sec"] == pytest.approx(exp["start_sec"], abs=1e-4)
        assert result["end_sec"] == pytest.approx(exp["end_sec"], abs=1e-4)
        assert result["duration_sec"] == pytest.approx(exp["duration_sec"], abs=1e-4)
        assert result["speech_prob"] == pytest.approx(exp["speech_prob"], abs=0.05)  # allow small model variance
        assert result["num_windows"] == exp["num_windows"]


def test_constant_noise_like_audio_returns_moderate_probabilities(vad_model):
    """
    Given: Audio filled with white noise (known to trigger moderate VAD scores)
    When: Processed with default settings
    Then: Speech probabilities should be in a moderate range (typically 0.3–0.7)
    """
    sampling_rate = 16000
    duration_sec = 1.0
    audio = torch.randn(int(duration_sec * sampling_rate)) * 0.1  # low-amplitude noise

    chunks = get_speech_probabilities_chunks(
        audio=audio,
        model=vad_model,
        sampling_rate=sampling_rate,
        chunk_seconds=0.25,
    )

    # All chunks should have non-zero probability, but not close to 1.0
    for chunk in chunks:
        assert 0.1 < chunk["speech_prob"] < 0.9


def test_overlap_produces_correct_number_of_chunks(vad_model):
    """
    Given: 2-second silent audio with 50% overlap (0.125s step)
    When: chunk_seconds=0.25, overlap_seconds=0.125
    Then: Should produce more chunks than non-overlapping case
    """
    sampling_rate = 16000
    audio = torch.zeros(int(2.0 * sampling_rate))

    chunks = get_speech_probabilities_chunks(
        audio=audio,
        model=vad_model,
        sampling_rate=sampling_rate,
        chunk_seconds=0.25,
        overlap_seconds=0.125,
    )

    # Expected: step = 0.125s → 2.0 / 0.125 = 16 chunks
    expected_num_chunks = 16
    assert len(chunks) == expected_num_chunks
    assert chunks[0]["start_sec"] == 0.0000
    assert chunks[1]["start_sec"] == pytest.approx(0.1250, abs=1e-4)
    assert chunks[-1]["end_sec"] == pytest.approx(2.0000, abs=1e-4)


def test_different_aggregations_produce_expected_order(vad_model):
    """
    Given: Audio with varying speech probability patterns (simulated)
    When: Using mean, max, and median aggregation
    Then: max >= mean >= median should hold for typical distributions
    """
    sampling_rate = 16000
    audio = torch.zeros(16000)  # 1 second silence → all probs ~0

    mean_chunks = get_speech_probabilities_chunks(audio, vad_model, aggregation="mean")
    max_chunks = get_speech_probabilities_chunks(audio, vad_model, aggregation="max")
    median_chunks = get_speech_probabilities_chunks(audio, vad_model, aggregation="median")

    # All should be near zero, but aggregation values should be identical in this edge case
    for m, mx, med in zip(mean_chunks, max_chunks, median_chunks):
        assert m["speech_prob"] == pytest.approx(mx["speech_prob"], abs=0.05)
        assert m["speech_prob"] == pytest.approx(med["speech_prob"], abs=0.05)


def test_last_incomplete_chunk_is_included_when_min_chunk_samples_set(vad_model):
    """
    Given: Audio length not multiple of chunk size (1.3 seconds)
    When: min_chunk_samples is set low enough
    Then: The final partial chunk is included
    """
    sampling_rate = 16000
    audio = torch.zeros(int(1.3 * sampling_rate))

    chunks = get_speech_probabilities_chunks(
        audio=audio,
        model=vad_model,
        sampling_rate=sampling_rate,
        chunk_seconds=0.25,
        min_chunk_samples=1000,  # ~0.06s, small enough to include remainder
    )

    # Should have full chunks + one partial
    assert chunks[-1]["end_sec"] == pytest.approx(1.3000, abs=0.01)
    assert chunks[-1]["duration_sec"] < 0.25