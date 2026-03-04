# tests/test_speech_segmentor.py
"""
Unit tests for LiveSpeechSegmentor

Focuses on decision logic: start / partial send / final send / discard / pre-roll / reset
"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from jet.audio.speech.speechbrain.speech_segmentor import LiveSpeechSegmentor

# ── Test configuration ────────────────────────────────────────────────────────


@dataclass
class TestConfig:
    sample_rate: int = 16000
    chunk_samples: int = 512  # 32 ms @ 16 kHz
    chunk_duration_sec: float = 6.0
    min_speech_duration_sec: float = 0.4
    max_speech_duration_sec: float = 15.0
    min_silence_duration_sec: float = 0.5
    pre_roll_buffer_maxlen: int = 8  # ~256 ms pre-roll

    @property
    def chunk_bytes(self) -> int:
        return self.chunk_samples * 2

    @property
    def chunk_sec(self) -> float:
        return self.chunk_samples / self.sample_rate


@pytest.fixture
def cfg() -> TestConfig:
    return TestConfig()


@pytest.fixture
def send_mock() -> MagicMock:
    return MagicMock()


@pytest.fixture
def segmentor(cfg: TestConfig, send_mock: MagicMock) -> LiveSpeechSegmentor:
    return LiveSpeechSegmentor(
        config=cfg,
        sample_rate=cfg.sample_rate,
        on_send=send_mock,
        pre_roll_buffer_maxlen=cfg.pre_roll_buffer_maxlen,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def silent_chunk() -> bytes:
    return b"\x00\x00" * 512


def speech_chunk() -> bytes:
    # arbitrary non-zero data — value doesn't matter for these tests
    return b"\x12\x34" * 512


def advance_time(mock_time: Any, seconds: float) -> float:
    """Helper to advance mocked monotonic time"""
    mock_time.return_value += seconds
    return mock_time.return_value


class TestLiveSpeechSegmentor:
    # --------------------------------------------------------------------------
    #  No activity
    # --------------------------------------------------------------------------

    def test_silence_does_not_start_segment(
        self, segmentor: LiveSpeechSegmentor, send_mock: MagicMock
    ):
        # Given a fresh segmentor
        # When we feed only silence for a long time
        for _ in range(120):
            segmentor.add_chunk(
                silent_chunk(),
                is_speech=False,
                speech_prob=0.08,
                rms=0.001,
            )

        # Then
        assert segmentor.current is None
        assert send_mock.call_count == 0

    # --------------------------------------------------------------------------
    #  Short utterance → discarded
    # --------------------------------------------------------------------------

    def test_short_speech_followed_by_silence_is_discarded(
        self, segmentor: LiveSpeechSegmentor, send_mock: MagicMock, cfg: TestConfig
    ):
        # Given we start speaking briefly
        for _ in range(10):  # ≈ 320 ms
            segmentor.add_chunk(
                speech_chunk(),
                is_speech=True,
                speech_prob=0.92,
                rms=0.12,
            )

        # Simulate progression of wallclock time for silence
        import time

        now = time.monotonic()
        for i in range(30):
            now += cfg.chunk_sec * 1.2  # slightly more than real chunk time
            segmentor.add_chunk(
                silent_chunk(),
                is_speech=False,
                speech_prob=0.06,
                rms=0.001,
                wallclock=now,
            )

        # Then short segment was discarded — no send occurred
        assert segmentor.current is None
        send_mock.assert_not_called()

    # --------------------------------------------------------------------------
    #  Normal utterance — long enough → final send on silence
    # --------------------------------------------------------------------------

    def test_normal_utterance_sent_as_final_on_silence(
        self, segmentor: LiveSpeechSegmentor, send_mock: MagicMock, cfg: TestConfig
    ):
        with patch("time.monotonic") as mock_time:
            mock_time.return_value = 1000.0

            # Given speech longer than min duration
            for _ in range(25):  # ≈ 800 ms
                segmentor.add_chunk(
                    speech_chunk(),
                    is_speech=True,
                    speech_prob=0.88,
                    rms=0.09,
                    wallclock=mock_time.return_value,
                )
                advance_time(mock_time, cfg.chunk_sec)

            # When long silence follows
            silence_start = mock_time.return_value
            for i in range(30):
                segmentor.add_chunk(
                    silent_chunk(),
                    is_speech=False,
                    speech_prob=0.04,
                    rms=0.0005,
                    wallclock=advance_time(mock_time, cfg.chunk_sec),
                )

            # Then one final send occurred
            assert send_mock.call_count == 1

            call = send_mock.call_args_list[0]
            audio_bytes, is_final, start_time, duration_sec, chunk_idx, stats = call[0]

            assert is_final is True
            assert chunk_idx == 0
            assert start_time == pytest.approx(1000.0)
            assert duration_sec >= cfg.min_speech_duration_sec
            assert stats["has_data"] is True

    # --------------------------------------------------------------------------
    #  Long utterance — multiple partial sends
    # --------------------------------------------------------------------------

    def test_long_utterance_sends_multiple_partials(
        self, segmentor: LiveSpeechSegmentor, send_mock: MagicMock, cfg: TestConfig
    ):
        with patch("time.monotonic") as mock_time:
            mock_time.return_value = 2000.0

            # Given continuous speech for ~22 seconds
            total_chunks = 0
            while total_chunks < 680:  # ~21.76 s
                segmentor.add_chunk(
                    speech_chunk(),
                    is_speech=True,
                    speech_prob=0.91,
                    rms=0.11,
                    wallclock=mock_time.return_value,
                )
                advance_time(mock_time, cfg.chunk_sec)
                total_chunks += 1

            # Then we should have several partial sends (every ~6 s)
            expected_partials = total_chunks * cfg.chunk_sec // cfg.chunk_duration_sec
            assert send_mock.call_count >= expected_partials - 1  # conservative

            # Check that most calls were partial
            calls = send_mock.call_args_list
            is_finals = [args[0][1] for args in calls]
            assert is_finals.count(True) == 0  # no final yet — still speaking
            assert is_finals.count(False) >= 3

    # --------------------------------------------------------------------------
    #  Max duration trim + partial send
    # --------------------------------------------------------------------------

    def test_exceeding_max_duration_trims_and_sends_partial(
        self, segmentor: LiveSpeechSegmentor, send_mock: MagicMock, cfg: TestConfig
    ):
        with patch("time.monotonic") as mock_time:
            mock_time.return_value = 3000.0

            # Given very long continuous speech
            chunks_sent = 0
            total_duration = 0.0

            while total_duration < cfg.max_speech_duration_sec + 4.0:
                segmentor.add_chunk(
                    speech_chunk(),
                    is_speech=True,
                    speech_prob=0.95,
                    rms=0.15,
                    wallclock=mock_time.return_value,
                )
                total_duration += cfg.chunk_sec
                advance_time(mock_time, cfg.chunk_sec)

                # Count how many times we sent partial
                if send_mock.call_count > chunks_sent:
                    chunks_sent = send_mock.call_count
                    # After first trim we expect duration close to max
                    if chunks_sent >= 1:
                        last_call = send_mock.call_args_list[-1]
                        sent_duration = last_call[0][3]
                        assert sent_duration <= cfg.max_speech_duration_sec + 0.1

    # --------------------------------------------------------------------------
    #  Pre-roll inclusion
    # --------------------------------------------------------------------------

    def test_pre_roll_is_included_in_first_segment(
        self, segmentor: LiveSpeechSegmentor, send_mock: MagicMock, cfg: TestConfig
    ):
        # Given we fill pre-roll with silent chunks first
        for _ in range(cfg.pre_roll_buffer_maxlen + 3):
            segmentor.add_chunk(
                silent_chunk(),
                is_speech=False,
                speech_prob=0.05,
                rms=0.0002,
            )

        # Simulate progression of wallclock time for speech and silence
        import time

        now = time.monotonic()
        # When speech starts
        for i in range(18):  # ~576 ms
            now += cfg.chunk_sec
            segmentor.add_chunk(
                speech_chunk(),
                is_speech=True,
                speech_prob=0.89,
                rms=0.10,
                wallclock=now,
            )

        # Give time for silence detection if needed (optional if test expects send during speech)
        for i in range(25):
            now += cfg.chunk_sec
            segmentor.add_chunk(
                silent_chunk(),
                is_speech=False,
                speech_prob=0.04,
                rms=0.001,
                wallclock=now,
            )

        # Then first send should include pre-roll silence
        assert send_mock.call_count >= 1
        first_send = send_mock.call_args_list[0]
        audio_bytes: bytes = first_send[0][0]

        # Rough check — should be longer than just the speech chunks
        expected_min_bytes = 18 * cfg.chunk_bytes
        assert len(audio_bytes) > expected_min_bytes

    # --------------------------------------------------------------------------
    #  Reset after final send
    # --------------------------------------------------------------------------

    def test_reset_after_final_send(
        self, segmentor: LiveSpeechSegmentor, send_mock: MagicMock, cfg: TestConfig
    ):
        with patch("time.monotonic") as mock_time:
            mock_time.return_value = 4000.0

            # Given a complete utterance
            for _ in range(22):
                segmentor.add_chunk(speech_chunk(), True, 0.9, 0.1)

            for _ in range(25):
                segmentor.add_chunk(silent_chunk(), False, 0.04, 0.001)
                advance_time(mock_time, cfg.chunk_sec)

            assert send_mock.call_count == 1
            assert segmentor.current is None
            assert segmentor.chunk_index == 0
            assert segmentor.last_send_time is None
            assert segmentor.silence_start is None

    # --------------------------------------------------------------------------
    #  Edge: exactly min duration speech → should send
    # --------------------------------------------------------------------------

    def test_speech_exactly_at_min_duration_is_sent(
        self, segmentor: LiveSpeechSegmentor, send_mock: MagicMock, cfg: TestConfig
    ):
        with patch("time.monotonic") as mock_time:
            mock_time.return_value = 5000.0

            min_chunks = int(cfg.min_speech_duration_sec / cfg.chunk_sec) + 1

            for _ in range(min_chunks):
                segmentor.add_chunk(
                    speech_chunk(),
                    True,
                    0.87,
                    0.08,
                    wallclock=mock_time.return_value,
                )
                advance_time(mock_time, cfg.chunk_sec)

            import time

            now = time.monotonic()  # or continue from previous now if you track it
            for _ in range(20):
                now += cfg.chunk_sec * 1.1
                segmentor.add_chunk(
                    silent_chunk(),
                    False,
                    0.03,
                    0.0008,
                    wallclock=now,
                )

            assert send_mock.call_count == 1
            assert send_mock.call_args[0][1] is True  # is_final
