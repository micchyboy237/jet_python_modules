# jet_python_modules/jet/audio/audio_waveform/tests/test_speech_tracker.py
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fireredvad.core.stream_vad_postprocessor import StreamVadFrameResult
from jet.audio.audio_waveform.speech_tracker import StreamingSpeechTracker


@pytest.fixture
def tmp_save_dir(tmp_path: Path) -> Path:
    return tmp_path / "segments"


@pytest.fixture
def mock_vad() -> MagicMock:
    vad = MagicMock()
    vad.detect_chunk.return_value = []
    vad.reset.return_value = None
    return vad


def test_init_creates_directory_and_sets_defaults(tmp_save_dir: Path):
    tracker = StreamingSpeechTracker(str(tmp_save_dir))
    assert tracker.save_dir.exists()
    assert tracker.min_speech_duration_sec == 0.3
    assert tracker.min_silence_duration_sec == 0.2
    assert tracker.max_speech_duration_sec == 10.0
    assert tracker.min_speech_samples == int(0.3 * tracker.sample_rate)
    assert tracker.max_speech_samples == int(10.0 * tracker.sample_rate)
    assert tracker.segment_counter == 0


def test_init_uses_custom_durations(tmp_save_dir: Path):
    tracker = StreamingSpeechTracker(
        str(tmp_save_dir),
        min_speech_duration_sec=0.5,
        min_silence_duration_sec=0.15,
        max_speech_duration_sec=8.0,
    )
    assert tracker.min_speech_duration_sec == 0.5
    assert tracker.min_silence_duration_sec == 0.15
    assert tracker.max_speech_duration_sec == 8.0


def test_init_accepts_injected_vad_for_testing(tmp_save_dir: Path, mock_vad: MagicMock):
    tracker = StreamingSpeechTracker(str(tmp_save_dir), vad=mock_vad)
    assert tracker.vad is mock_vad


def test_process_chunk_no_speech_does_nothing(tmp_save_dir: Path, mock_vad: MagicMock):
    tracker = StreamingSpeechTracker(str(tmp_save_dir), vad=mock_vad)
    tracker.process_chunk(np.zeros(512, dtype=np.float32))
    assert not tracker.is_speaking
    assert len(tracker.speech_buffer) == 0


def test_full_speech_cycle_saves_segment(tmp_save_dir: Path):
    tracker = StreamingSpeechTracker(
        str(tmp_save_dir), min_speech_duration_sec=0.3, max_speech_duration_sec=10.0
    )
    start_result = StreamVadFrameResult(
        frame_idx=10,
        is_speech=True,
        raw_prob=0.8,
        smoothed_prob=0.8,
        is_speech_start=True,
        speech_start_frame=5,
    )
    end_result = StreamVadFrameResult(
        frame_idx=80,
        is_speech=False,
        raw_prob=0.2,
        smoothed_prob=0.2,
        is_speech_end=True,
        speech_end_frame=75,
    )
    mock_vad = MagicMock()
    mock_vad.detect_chunk.side_effect = [[start_result], [end_result]]
    tracker.vad = mock_vad
    speech_audio = np.random.rand(8000).astype(np.float32)
    tracker.process_chunk(speech_audio[:4000])
    tracker.process_chunk(speech_audio[4000:])
    seg_dir = tmp_save_dir / "segment_0001"
    assert seg_dir.exists()
    assert (seg_dir / "sound.wav").exists()
    assert (seg_dir / "segment.json").exists()
    assert (seg_dir / "speech_probs.json").exists()
    with open(seg_dir / "segment.json") as f:
        meta = json.load(f)
        assert meta["duration_sec"] >= 0.3
        assert meta["max_speech_duration_sec"] == 10.0
        assert "prob_info" in meta
        pinfo = meta["prob_info"]
        assert pinfo["num_frames"] == 2
        assert pinfo["avg_smoothed_prob"] == 0.5
        assert pinfo["min_smoothed_prob"] == 0.2
        assert pinfo["max_smoothed_prob"] == 0.8
    with open(seg_dir / "speech_probs.json") as f:
        probs = json.load(f)
        assert isinstance(probs, list)
        assert probs == [0.8, 0.2]


def test_short_speech_is_discarded(tmp_save_dir: Path):
    tracker = StreamingSpeechTracker(
        str(tmp_save_dir), min_speech_duration_sec=0.4, max_speech_duration_sec=10.0
    )
    start_result = StreamVadFrameResult(
        frame_idx=10,
        is_speech=True,
        raw_prob=0.8,
        smoothed_prob=0.8,
        is_speech_start=True,
        speech_start_frame=5,
    )
    end_result = StreamVadFrameResult(
        frame_idx=30,
        is_speech=False,
        raw_prob=0.2,
        smoothed_prob=0.2,
        is_speech_end=True,
        speech_end_frame=28,
    )
    mock_vad = MagicMock()
    mock_vad.detect_chunk.side_effect = [[start_result], [end_result]]
    tracker.vad = mock_vad
    short_audio = np.random.rand(3000).astype(np.float32)
    tracker.process_chunk(short_audio[:1500])
    tracker.process_chunk(short_audio[1500:])
    assert not any((tmp_save_dir / f"segment_{i:04d}").exists() for i in range(1, 10))


def test_max_speech_duration_forces_save(tmp_save_dir: Path):
    """Force a save when speech exceeds max_speech_duration_sec (0.5s in this test)."""
    tracker = StreamingSpeechTracker(
        str(tmp_save_dir),
        max_speech_duration_sec=0.5,
    )
    start_result = StreamVadFrameResult(
        frame_idx=10,
        is_speech=True,
        raw_prob=0.8,
        smoothed_prob=0.8,
        is_speech_start=True,
        speech_start_frame=5,
    )
    mock_vad = MagicMock()
    mock_vad.detect_chunk.side_effect = [[start_result]] + [[]] * 39
    tracker.vad = mock_vad
    long_audio = np.random.rand(20000).astype(np.float32)
    for i in range(0, len(long_audio), 600):
        tracker.process_chunk(long_audio[i : i + 600])
    seg_path = tmp_save_dir / "segment_0001" / "sound.wav"
    assert seg_path.exists(), "sound.wav was not created! (check [DEBUG] logs with -s)"
    with open(tmp_save_dir / "segment_0001" / "segment.json") as f:
        meta = json.load(f)
        assert meta["duration_sec"] >= 0.5
        assert meta["max_speech_duration_sec"] == 0.5


def test_multiple_segments_are_numbered_sequentially(tmp_save_dir: Path):
    """Two separate speech segments must create segment_0001 and segment_0002."""
    tracker = StreamingSpeechTracker(
        str(tmp_save_dir),
        min_speech_duration_sec=0.05,
    )
    start1 = StreamVadFrameResult(
        frame_idx=10,
        is_speech=True,
        raw_prob=0.8,
        smoothed_prob=0.8,
        is_speech_start=True,
        speech_start_frame=5,
    )
    end1 = StreamVadFrameResult(
        frame_idx=20,
        is_speech=False,
        raw_prob=0.2,
        smoothed_prob=0.2,
        is_speech_end=True,
        speech_end_frame=18,
    )
    start2 = StreamVadFrameResult(
        frame_idx=50,
        is_speech=True,
        raw_prob=0.8,
        smoothed_prob=0.8,
        is_speech_start=True,
        speech_start_frame=45,
    )
    end2 = StreamVadFrameResult(
        frame_idx=60,
        is_speech=False,
        raw_prob=0.2,
        smoothed_prob=0.2,
        is_speech_end=True,
        speech_end_frame=58,
    )
    mock_vad = MagicMock()
    mock_vad.detect_chunk.side_effect = [[start1], [end1], [start2], [end2]] + [[]] * 4
    tracker.vad = mock_vad
    for _ in range(8):
        tracker.process_chunk(np.random.rand(800).astype(np.float32))
    assert (tmp_save_dir / "segment_0001").exists()
    assert (tmp_save_dir / "segment_0002").exists()


def test_reset_clears_state(tmp_save_dir: Path, mock_vad: MagicMock):
    tracker = StreamingSpeechTracker(str(tmp_save_dir), vad=mock_vad)
    tracker.is_speaking = True
    tracker.speech_buffer = np.ones(1000, dtype=np.float32)
    tracker.segment_counter = 5
    tracker.reset()
    assert tracker.is_speaking is False
    assert len(tracker.speech_buffer) == 0
    assert tracker.segment_counter == 0
    mock_vad.reset.assert_called_once()


def test_long_speech_vad_split_saves_multiple_segments_without_loss(tmp_save_dir: Path):
    """VAD max-split (end + start in same chunk) must produce two segments;
    no audio after the split point is lost."""
    tracker = StreamingSpeechTracker(str(tmp_save_dir), max_speech_duration_sec=0.5)
    start1 = StreamVadFrameResult(
        frame_idx=10,
        is_speech=True,
        raw_prob=0.8,
        smoothed_prob=0.8,
        is_speech_start=True,
        speech_start_frame=5,
    )
    split_chunk = [
        StreamVadFrameResult(
            frame_idx=55,
            is_speech=False,
            raw_prob=0.2,
            smoothed_prob=0.2,
            is_speech_end=True,
            speech_end_frame=55,
        ),
        StreamVadFrameResult(
            frame_idx=56,
            is_speech=True,
            raw_prob=0.8,
            smoothed_prob=0.8,
            is_speech_start=True,
            speech_start_frame=56,
        ),
    ]
    end2 = StreamVadFrameResult(
        frame_idx=110,
        is_speech=False,
        raw_prob=0.2,
        smoothed_prob=0.2,
        is_speech_end=True,
        speech_end_frame=110,
    )
    mock_vad = MagicMock()
    mock_vad.detect_chunk.side_effect = (
        [[start1]] + [[]] * 20 + [split_chunk] + [[]] * 10 + [[end2]]
    )
    tracker.vad = mock_vad

    audio = np.random.rand(30000).astype(np.float32)
    for i in range(0, len(audio), 600):
        tracker.process_chunk(audio[i : i + 600])

    seg1 = tmp_save_dir / "segment_0001"
    seg2 = tmp_save_dir / "segment_0002"
    assert seg1.exists() and (seg1 / "sound.wav").exists()
    assert seg2.exists() and (seg2 / "sound.wav").exists()
    # both segments received their audio (no loss after split)
    meta1 = json.loads((seg1 / "segment.json").read_text())
    meta2 = json.loads((seg2 / "segment.json").read_text())
    assert meta1["duration_sec"] >= 0.5
    assert meta2["duration_sec"] >= 0.3  # second part also saved
