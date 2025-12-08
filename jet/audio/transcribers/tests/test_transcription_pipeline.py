# tests/test_transcription_pipeline.py
from __future__ import annotations

from collections import deque
import time

import numpy as np
import pytest
from numpy.typing import NDArray

from jet.audio.transcribers.transcription_pipeline import AudioKey, TranscriptionPipeline


# ----------------------------------------------------------------------
# Fixtures & test doubles
# ----------------------------------------------------------------------
@pytest.fixture
def silent_audio_1sec() -> NDArray[np.float32]:
    """1 second of pure silence at 16 kHz."""
    return np.zeros(16_000, dtype=np.float32)


@pytest.fixture
def chirp_audio_0_5sec() -> NDArray[np.float32]:
    """Short unique chirp – perfect for cache testing."""
    sr = 16_000
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
    chirp = np.sin(2 * np.pi * 1000 * t ** 2) * 0.3
    return chirp.astype(np.float32)


@pytest.fixture
def pipeline() -> TranscriptionPipeline:
    """Small pipeline – fast executor + tiny cache for deterministic tests."""
    return TranscriptionPipeline(max_workers=4, cache_size=10)


# ----------------------------------------------------------------------
# Helper – drain the internal queue (only used in tests)
# ----------------------------------------------------------------------
def _drain_pipeline(pipeline: TranscriptionPipeline, timeout: float = 3.0) -> None:
    start = time.time()
    while pipeline._queue:  # type: ignore[attr-defined]
        if time.time() - start > timeout:
            raise TimeoutError("Pipeline did not finish in time")
        time.sleep(0.01)


# ----------------------------------------------------------------------
# Tests (BDD style)
# ----------------------------------------------------------------------
def test_audio_key_is_hashable_and_immutable(silent_audio_1sec: NDArray[np.float32]):
    # Given
    key1 = AudioKey(hash=123, duration_sec=1.0)
    key2 = AudioKey(hash=123, duration_sec=1.0)
    key3 = AudioKey(hash=999, duration_sec=1.0)

    # When / Then
    assert key1 == key2
    assert key1 != key3
    assert hash(key1) == hash(key2)
    assert len({key1, key2, key3}) == 2

    # Immutability
    with pytest.raises(AttributeError):
        key1.hash = 456  # type: ignore[misc]


def test_cache_hit_returns_immediately_without_calling_functions(
    pipeline: TranscriptionPipeline,
    silent_audio_1sec: NDArray[np.float32],
    mocker,
):
    # Given – manually populate cache
    fake_ja = "こんにちは"
    fake_en = "Hello"
    key = pipeline._make_key(silent_audio_1sec)
    pipeline._cache_set(key, fake_ja, fake_en)

    process_spy = mocker.spy(pipeline, "_process")

    # When
    pipeline.submit_segment(silent_audio_1sec)

    # Then
    assert len(pipeline._queue) == 0          # no work submitted
    process_spy.assert_not_called()


def test_identical_audio_hits_cache_on_second_submission(
    pipeline: TranscriptionPipeline,
    chirp_audio_0_5sec: NDArray[np.float32],
    mocker,
):
    # Given
    transcribe_mock = mocker.patch(
        "jet.audio.transcribers.transcription_pipeline.transcribe_ja_chunk", return_value="テスト"
    )
    translate_mock = mocker.patch(
        "jet.audio.transcribers.transcription_pipeline.translate_ja_en", return_value="Test"
    )

    # When – first call (cache miss)
    pipeline.submit_segment(chirp_audio_0_5sec)
    _drain_pipeline(pipeline)

    # When – second identical call
    pipeline.submit_segment(chirp_audio_0_5sec.copy())

    # Then – heavy functions called only once
    transcribe_mock.assert_called_once()
    translate_mock.assert_called_once()
    assert len(pipeline._queue) == 0


def test_different_audio_creates_separate_cache_entries(
    pipeline: TranscriptionPipeline,
    silent_audio_1sec: NDArray[np.float32],
    chirp_audio_0_5sec: NDArray[np.float32],
    mocker,
):
    # Given
    transcribe_mock = mocker.patch("jet.audio.transcribers.transcription_pipeline.transcribe_ja_chunk")
    translate_mock = mocker.patch("jet.audio.transcribers.transcription_pipeline.translate_ja_en")

    # When
    pipeline.submit_segment(silent_audio_1sec)
    pipeline.submit_segment(chirp_audio_0_5sec)

    _drain_pipeline(pipeline)

    # Then
    assert transcribe_mock.call_count == 2
    assert translate_mock.call_count == 2


def test_lru_eviction_when_cache_is_full(pipeline: TranscriptionPipeline, mocker):
    # ← THIS IS THE ONLY CORRECT WAY
    with pipeline._lock:
        pipeline._cache.clear()
        # Completely replace the order deque with a new one of maxlen=3
        pipeline._cache_order = deque(maxlen=3)

    # Make 100% sure each audio has a unique hash
    audios = [
        np.random.randn(8_000).astype(np.float32) + float(i) * 1000.0  # huge offset → guaranteed unique bytes
        for i in range(10)
    ]

    mocker.patch(
        "jet.audio.transcribers.transcription_pipeline.transcribe_ja_chunk",
        side_effect=lambda x: f"text_{id(x)}",
    )
    mocker.patch(
        "jet.audio.transcribers.transcription_pipeline.translate_ja_en",
        return_value="en_translation",
    )

    for audio in audios:
        pipeline.submit_segment(audio)

    _drain_pipeline(pipeline)

    cache_keys = list(pipeline._cache.keys())
    print(f"Final cache keys: {cache_keys}")  # optional debug

    assert len(pipeline._cache) == 3
    assert pipeline._make_key(audios[0]) not in pipeline._cache
    assert pipeline._make_key(audios[-1]) in pipeline._cache
    # The three most recent should be in cache
    assert pipeline._make_key(audios[-3]) in pipeline._cache
    assert pipeline._make_key(audios[-2]) in pipeline._cache
    assert pipeline._make_key(audios[-1]) in pipeline._cache


def test_exception_in_worker_is_caught_and_logged(
    pipeline: TranscriptionPipeline,
    silent_audio_1sec: NDArray[np.float32],
    mocker,
):
    # Given
    mocker.patch(
        "jet.audio.transcribers.transcription_pipeline.transcribe_ja_chunk",
        side_effect=RuntimeError("Model crashed!"),
    )
    console_spy = mocker.patch("jet.audio.transcribers.transcription_pipeline.console.print")

    # When
    pipeline.submit_segment(silent_audio_1sec)
    _drain_pipeline(pipeline)

    # Then
    console_spy.assert_called_once()
    call_args = " ".join(console_spy.call_args[0])
    assert "Transcription pipeline error" in call_args


def test_make_key_is_deterministic_and_rounds_duration(
    pipeline: TranscriptionPipeline,
    silent_audio_1sec: NDArray[np.float32],
):
    # Given
    audio1 = silent_audio_1sec
    audio2 = np.pad(silent_audio_1sec, (0, 3))  # slightly longer

    # When
    key1 = pipeline._make_key(audio1)
    key2 = pipeline._make_key(audio1.copy())
    key3 = pipeline._make_key(audio2)

    # Then
    assert key1 == key2
    assert key1 != key3
    assert key1.duration_sec == 1.0
    assert key3.duration_sec == pytest.approx(1.000, abs=0.001)


def test_shutdown_wait_drains_all_tasks(pipeline: TranscriptionPipeline, mocker):
    # Given – slow transcribe
    mocker.patch(
        "jet.audio.transcribers.transcription_pipeline.transcribe_ja_chunk",
        side_effect=lambda _: time.sleep(0.15),
    )

    # When
    for _ in range(6):
        pipeline.submit_segment(np.zeros(16_000, dtype=np.float32))

    start = time.time()
    pipeline.shutdown(wait=True)
    elapsed = time.time() - start

    # Then
    assert elapsed >= 0.2  # some real work happened


def test_shutdown_nowait_returns_immediately(pipeline: TranscriptionPipeline, mocker):
    # Given – transcribe that never finishes
    forever_mock = mocker.patch(
        "jet.audio.transcribers.transcription_pipeline.transcribe_ja_chunk",
        side_effect=lambda _: time.sleep(10),
    )

    # When
    pipeline.submit_segment(np.zeros(16_000, dtype=np.float32))
    pipeline.shutdown(wait=False)

    # Then – returns instantly, task is cancelled
    time.sleep(0.05)
    assert forever_mock.call_count == 1  # was invoked
    assert len(pipeline._queue) == 0      # queue cleared