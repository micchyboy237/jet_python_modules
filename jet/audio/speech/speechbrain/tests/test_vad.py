"""
Unit tests for SpeechBrainVAD streaming wrapper.

Uses heavy mocking to avoid real model loading and torchaudio file I/O.
Focuses on buffer management, caching, consistency with offline path.
"""

import time
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from jet.audio.speech.speechbrain.vad import SpeechBrainVAD

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_vad() -> Generator[SpeechBrainVAD, None, None]:
    """Mocked SpeechBrainVAD instance – no real model loading."""
    with patch("speechbrain.inference.VAD.VAD.from_hparams") as mock_from_hparams:
        mock_vad_instance = MagicMock()
        mock_from_hparams.return_value = mock_vad_instance

        vad = SpeechBrainVAD(
            max_history_sec=3.0,
            sample_rate=16000,
            device="cpu",
            overlap_small_chunk=True,
        )

        # Fake fallback single-chunk inference
        vad.vad.get_speech_prob_chunk.return_value = torch.tensor([[[0.42]]])

        yield vad


@pytest.fixture
def silence_chunk_10ms() -> np.ndarray:
    """10 ms of silence at 16 kHz."""
    return np.zeros(160, dtype=np.float32)


@pytest.fixture
def noise_chunk_10ms() -> np.ndarray:
    """10 ms of white noise at 16 kHz."""
    return np.random.randn(160).astype(np.float32) * 0.1


@pytest.fixture
def speech_like_chunk_160ms() -> np.ndarray:
    """160 ms of fake speech-like signal."""
    t = np.linspace(0, 0.16, 2560, endpoint=False)
    signal = 0.8 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    return signal.astype(np.float32)


@pytest.fixture
def fake_extract_side_effect(mock_vad):
    """Reusable side effect: returns realistic (segments, probs) based on buffer length."""

    def side_effect(*args, **kwargs):
        n_samples = len(mock_vad.audio_history)
        n_frames = max(1, (n_samples + 159) // 160)  # approximate 10 ms hop
        fake_segments = (
            [] if n_samples < 400 else [{"start": 0.0, "end": 0.1, "prob": 0.7}]
        )
        fake_probs = [0.0 if i < 5 else 0.85 for i in range(n_frames)]
        return fake_segments, fake_probs

    return side_effect


# ── Tests ───────────────────────────────────────────────────────────────────


def test_init_calculates_correct_buffer_size(mock_vad: SpeechBrainVAD):
    """Given a 3-second history limit, buffer maxlen should match samples."""
    # Given: SpeechBrainVAD initialized with max_history_sec=3.0 and 16000 Hz
    # When: we check the internal deque maxlen
    # Then: it should be exactly 48000 samples
    assert mock_vad.max_history_samples == 48000
    assert mock_vad.audio_history.maxlen == 48000


def test_append_audio_converts_to_mono_and_float32(mock_vad: SpeechBrainVAD):
    """Given stereo or int16 input, audio is converted correctly."""
    # Given: a stereo int16 chunk
    stereo_int = np.random.randint(-32768, 32767, (2, 320), dtype=np.int16)

    # When: we append it
    mock_vad.append_audio(stereo_int)

    # Then: history contains normalized float32 mono data in ≈[-1,1] range
    assert len(mock_vad.audio_history) == 320
    samples = np.array(mock_vad.audio_history)
    assert samples.dtype == np.float32
    assert samples.min() >= -1.0 - 1e-6  # small epsilon for float precision
    assert samples.max() <= 1.0 + 1e-6


def test_append_respects_maxlen_and_overwrites_oldest(mock_vad: SpeechBrainVAD):
    """Given buffer fills beyond limit, oldest samples are dropped."""
    # Given: vad with 3s history (48000 samples)
    chunk = np.ones(16000, dtype=np.float32) * 0.7  # 1 second

    # When: we append 5 seconds of audio (5 chunks)
    for _ in range(5):
        mock_vad.append_audio(chunk)

    # Then: buffer length = 48000, oldest data overwritten
    assert len(mock_vad.audio_history) == 48000
    samples = np.array(mock_vad.audio_history)
    assert np.all(samples[:16000] == 0.7)  # first 1s of last appended chunks
    assert np.all(samples[-16000:] == 0.7)  # last 1s


def test_get_latest_prob_fallback_when_no_cache(mock_vad: SpeechBrainVAD):
    """Given no probs cached yet, fallback to single chunk inference."""
    # Given: empty history + some audio
    mock_vad.append_audio(np.random.randn(400).astype(np.float32))

    # When: we call get_latest_prob()
    prob = mock_vad.get_latest_prob()

    # Then: it uses vad.get_speech_prob_chunk and returns scalar float
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    mock_vad.vad.get_speech_prob_chunk.assert_called_once()


def test_get_frame_probs_returns_cached_when_not_forced(
    mock_vad: SpeechBrainVAD, mocker
):
    """Given recent cache, get_frame_probs returns it without recompute."""
    # Given: probs already cached 0.5 seconds ago
    fake_probs = [0.1, 0.2, 0.3, 0.9, 0.95]
    mock_vad.current_probs = fake_probs[:]
    mock_vad.last_full_recompute_time = time.time() - 0.5

    extract_mock = mocker.patch(
        "jet.audio.speech.speechbrain.vad.extract_speech_timestamps",
        return_value=([], [0.1] * 5),
    )

    # When: we call get_frame_probs() without force
    probs = mock_vad.get_frame_probs(force_recompute=False)

    # Then: cached list is returned, no file I/O or extraction called
    assert probs == fake_probs
    extract_mock.assert_not_called()


def test_get_frame_probs_force_recompute_calls_extractor(
    mock_vad: SpeechBrainVAD, mocker, fake_extract_side_effect
):
    """Given force_recompute=True, runs full extraction pipeline."""
    # Given: some audio in buffer
    mock_vad.append_audio(np.random.randn(3200).astype(np.float32))

    extract_mock = mocker.patch(
        "jet.audio.speech.speechbrain.vad.extract_speech_timestamps",
        side_effect=fake_extract_side_effect,
    )

    # When: we force recompute probs
    probs = mock_vad.get_frame_probs(force_recompute=True)

    # Then: extract_speech_timestamps is called, probs cached and returned
    assert probs == mock_vad.current_probs
    assert len(probs) > 0
    extract_mock.assert_called_once()


def test_get_frame_probs_short_buffer_returns_zeros(mock_vad: SpeechBrainVAD):
    """Given very short audio (< ~10 ms), returns zero probs."""
    # Given: only 80 samples in buffer
    mock_vad.append_audio(np.zeros(80, dtype=np.float32))

    # When: we request frame probs
    probs = mock_vad.get_frame_probs(force_recompute=True)

    # Then: length matches buffer, all zeros
    assert len(probs) == 80
    assert all(p == 0.0 for p in probs)


def test_get_speech_segments_short_buffer_returns_empty_list(mock_vad: SpeechBrainVAD):
    """Given buffer < ~25 ms, no segments returned."""
    # Given: very short audio
    mock_vad.append_audio(np.random.randn(300).astype(np.float32))

    # When: we ask for segments
    segments = mock_vad.get_speech_segments()

    # Then: empty list returned
    assert segments == []


def test_reset_clears_all_state(
    mock_vad: SpeechBrainVAD, mocker, fake_extract_side_effect
):
    """Given populated buffer & cache, reset clears everything."""
    # Given: filled buffer and cached probs
    mock_vad.append_audio(np.ones(8000, dtype=np.float32))

    mocker.patch(
        "jet.audio.speech.speechbrain.vad.extract_speech_timestamps",
        side_effect=fake_extract_side_effect,
    )

    mock_vad.get_frame_probs(force_recompute=True)

    assert len(mock_vad.audio_history) > 0
    assert len(mock_vad.current_probs) > 0

    # When: we call reset()
    mock_vad.reset()

    # Then: both buffer and probs are empty, timestamp reset
    assert len(mock_vad.audio_history) == 0
    assert len(mock_vad.current_probs) == 0
    assert mock_vad.last_full_recompute_time == 0.0


def test_append_and_get_probs_length_matches_expected_frames(
    mock_vad: SpeechBrainVAD, mocker, fake_extract_side_effect
):
    """Given 1.2 seconds of audio, probs length ≈ 120 (10 ms hop)."""
    # Given: 1200 ms of continuous audio appended
    chunk = np.sin(2 * np.pi * 440 * np.linspace(0, 0.01, 160))
    for _ in range(120):
        mock_vad.append_audio(chunk.astype(np.float32))

    mocker.patch(
        "jet.audio.speech.speechbrain.vad.extract_speech_timestamps",
        side_effect=fake_extract_side_effect,
    )

    # When: we force compute full probs
    probs = mock_vad.get_frame_probs(force_recompute=True)

    # Then: number of probs ≈ total_samples / 160 (± a few due to padding/alignment)
    total_samples = 120 * 160
    expected_approx = total_samples // 160
    assert abs(len(probs) - expected_approx) <= 15
