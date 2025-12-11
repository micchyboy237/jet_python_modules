# tests/test_transcription_pipeline.py

import logging
from concurrent.futures import Future

import numpy as np
import pytest
from numpy.typing import NDArray

# Correct import for your project structure
from jet.audio.transcribers.transcription_pipeline import (
    TranscriptionPipeline,
)

# ----------------------------------------------------------------------
# Mock the real endpoint
# ----------------------------------------------------------------------
MOCK_TRANSCRIPTION = "こんにちは、世界"
MOCK_TRANSLATION = "Hello, world"
MOCK_WORDS = [
    {"word": "こんにちは", "start": 0.0, "end": 0.8},
    {"word": "、", "start": 0.8, "end": 0.9},
    {"word": "世界", "start": 0.9, "end": 1.5},
]

call_count = 0

def mock_transcribe_audio(_: bytes) -> dict:
    global call_count
    call_count += 1
    return {
        "transcription": MOCK_TRANSCRIPTION,
        "translation": MOCK_TRANSLATION,
        "words": MOCK_WORDS,
    }

@pytest.fixture(autouse=True)
def setup_mock(monkeypatch):
    global call_count
    call_count = 0
    monkeypatch.setattr("jet.audio.transcribers.transcription_pipeline.transcribe_audio", mock_transcribe_audio)

@pytest.fixture
def silent_audio() -> NDArray[np.float32]:
    return np.zeros(16000, dtype=np.float32)

@pytest.fixture
def different_audio() -> NDArray[np.float32]:
    audio = np.zeros(16000, dtype=np.float32)
    audio[::100] = 1.0
    return audio

# ----------------------------------------------------------------------
# Tests – Only updated/fixed versions below
# ----------------------------------------------------------------------

def test_lru_eviction_when_cache_full(
    silent_audio: NDArray[np.float32],
    different_audio: NDArray[np.float32],
):
    """
    Given: cache_size=1
    When: two different audio chunks are processed sequentially
    Then: the first one is evicted, only the second remains
    """
    pipeline = TranscriptionPipeline(cache_size=1, max_workers=1)

    pipeline.submit_segment(silent_audio)
    pipeline.submit_segment(different_audio)

    pipeline.shutdown(wait=True)

    key1 = pipeline._make_key(silent_audio)
    key2 = pipeline._make_key(different_audio)

    # First key must be evicted
    expected = None
    result = pipeline._cache_get(key1)
    assert result is expected, f"LRU eviction failed — old key still present: {result!r}"

    # Second key must be cached with the exact expected values (real Japanese text)
    expected = (
        "こんにちは、世界",           # ← real Japanese, not \u escapes
        "Hello, world",
        [
            {"word": "こんにちは", "start": 0.0, "end": 0.8},
            {"word": "、", "start": 0.8, "end": 0.9},
            {"word": "世界", "start": 0.9, "end": 1.5},
        ],
    )
    result = pipeline._cache_get(key2)
    assert result == expected, f"Latest result not cached correctly: {result!r}"

def test_print_result_is_called_and_looks_correct(
    silent_audio: NDArray[np.float32],
    capsys,
):
    # Given: pipeline
    pipeline = TranscriptionPipeline(max_workers=1)

    # When
    pipeline.submit_segment(silent_audio)
    pipeline.shutdown(wait=True)

    # Then: output contains expected text
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "Live Translation" in output
    assert MOCK_TRANSCRIPTION in output
    assert MOCK_TRANSLATION in output

def test_duplicate_concurrent_submissions_are_deduplicated(
    silent_audio: NDArray[np.float32],
    caplog,
):
    """
    Given: pipeline with multiple workers
    When: same audio submitted 10 times rapidly
    Then: only ONE real transcription occurs (even though cache miss may be logged >1)
    """
    pipeline = TranscriptionPipeline(max_workers=6, cache_size=50)

    with caplog.at_level(logging.DEBUG):
        for _ in range(10):
            pipeline.submit_segment(silent_audio)

        pipeline.shutdown(wait=True)

    # This is the only thing that truly matters:
    expected = 1
    result = call_count
    assert result == expected, f"Expected only 1 transcription call, but got {call_count}"

    # Log may show multiple "CACHE MISS" due to race – that’s expected with current code
    misses = sum(1 for r in caplog.records if "CACHE MISS" in r.message)
    assert misses >= 1

def test_shutdown_cancels_pending_futures_when_wait_false(
    silent_audio: NDArray[np.float32],
    mocker,
):
    # Given: mock slow task
    future = mocker.MagicMock(spec=Future)
    future.cancel.return_value = True
    future.cancelled.return_value = True

    executor = mocker.MagicMock()
    executor.submit.return_value = future

    pipeline = TranscriptionPipeline(max_workers=1)
    pipeline._executor = executor  # direct inject

    # submit
    pipeline.submit_segment(silent_audio)

    # When: shutdown without wait
    pipeline.shutdown(wait=False)

    # Then: cancel was called and queue cleared
    future.cancel.assert_called_once()
    assert len(pipeline._queue) == 0