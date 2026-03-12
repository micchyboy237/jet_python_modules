import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Assuming the class is in streaming_speech_tracker.py
from jet.audio.audio_waveform.speech_tracker2 import StreamingSpeechTracker


@pytest.fixture
def tracker(tmp_path: Path):
    save_dir = tmp_path / "vad_segments"
    return StreamingSpeechTracker(
        save_dir=str(save_dir),
        min_speech_duration_sec=0.3,
        min_silence_duration_sec=0.2,
        max_speech_duration_sec=2.0,
        sample_rate=16000,
        frame_shift_sec=0.01,
    )


def test_short_speech_ignored(tracker: StreamingSpeechTracker):
    # 25 frames speech = 0.25 s < 0.3 s → should be ignored
    for i in range(25):
        tracker.update(np.zeros(160, dtype=np.int16), is_speech=True, speech_prob=0.9)

    tracker.finalize()
    assert len(tracker.get_all_segments()) == 0
    assert len(list(tracker.save_dir.glob("segment_*"))) == 0


def test_min_speech_accepted(tracker: StreamingSpeechTracker):
    # 35 frames = 0.35 s > 0.3 s
    for i in range(35):
        tracker.update(
            np.ones(160, dtype=np.int16) * 1000, is_speech=True, speech_prob=0.92
        )

    tracker.finalize()
    segments = tracker.get_all_segments()
    assert len(segments) == 1
    assert segments[0].duration_sec >= 0.3

    folders = list(tracker.save_dir.glob("segment_*"))
    assert len(folders) == 1

    wav_file = folders[0] / "sound.wav"
    data, sr = sf.read(str(wav_file))
    assert sr == 16000
    assert len(data) > 0


def test_silence_closes_segment(tracker: StreamingSpeechTracker):
    # 40 frames speech → 20 silence → should close
    for _ in range(40):
        tracker.update(np.zeros(160), True, 0.88)

    for _ in range(20):
        tracker.update(np.zeros(160), False, 0.12)

    tracker.finalize()
    segments = tracker.get_all_segments()
    assert len(segments) == 1
    assert 0.35 < segments[0].duration_sec < 0.45


def test_max_duration_forces_split(tracker: StreamingSpeechTracker):
    # 250 frames speech = 2.5 s > max=2.0 s → should split
    for i in range(250):
        tracker.update(np.ones(160, dtype=np.int16) * (1000 + i % 100), True, 0.95)

    tracker.finalize()
    segments = tracker.get_all_segments()
    assert len(segments) >= 2
    assert all(s.duration_sec <= 2.1 for s in segments)


def test_summary_and_probs_files_written(tracker: StreamingSpeechTracker):
    for _ in range(45):
        tracker.update(np.zeros(160), True, 0.7 + 0.2 * (np.random.rand() - 0.5))

    for _ in range(25):
        tracker.update(np.zeros(160), False, 0.1)

    tracker.finalize()

    folders = list(tracker.save_dir.glob("segment_*"))
    assert len(folders) >= 1

    summary_file = folders[0] / "summary.json"
    probs_file = folders[0] / "speech_probs.json"

    assert summary_file.exists()
    assert probs_file.exists()

    with open(summary_file) as f:
        data = json.load(f)
        assert "start_sec" in data
        assert "end_sec" in data
        assert "duration_sec" in data
        assert data["duration_sec"] > 0

    with open(probs_file) as f:
        data = json.load(f)
        assert "probs" in data
        assert len(data["probs"]) > 20


if __name__ == "__main__":
    pytest.main(["-v", __file__])
