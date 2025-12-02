# test_silero_vad_stream.py (FIXED VERSION)

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Generator

import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
from pytest_mock import MockerFixture

from jet.audio.speech.silero.silero_vad_stream import SileroVADStreamer


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    dir_ = tmp_path / "segments"
    dir_.mkdir()
    return dir_


@pytest.fixture
def mock_audio_data() -> torch.Tensor:
    sr = 16000
    duration = 5.0
    num_samples = int(sr * duration)
    t = torch.linspace(0, duration, num_samples)

    silence = torch.randn(num_samples) * 0.01
    speech = 0.8 * torch.sin(2 * torch.pi * 1000 * t)
    mask = (t >= 1.5) & (t <= 3.5)
    audio = torch.where(mask, speech, silence)
    return audio


@pytest.fixture
def vad_streamer(output_dir: Path, mocker: MockerFixture) -> Generator[SileroVADStreamer, None, None]:
    mock_stream = mocker.MagicMock()
    mocker.patch("sounddevice.InputStream", return_value=mock_stream)

    streamer = SileroVADStreamer(
        threshold=0.5,
        sample_rate=16000,
        min_silence_duration_ms=500,
        speech_pad_ms=30,
        block_size=512,
        output_dir=output_dir,
        save_segments=True,
    )
    yield streamer
    shutil.rmtree(output_dir, ignore_errors=True)


# Helper to split audio into exact blocks (handles non-divisible lengths)
def _split_into_blocks(audio: torch.Tensor, block_size: int = 512):
    total_samples = len(audio)
    for start in range(0, total_samples, block_size):
        end = min(start + block_size, total_samples)
        block = audio[start:end]
        if len(block) < block_size:
            block = F.pad(block, (0, block_size - len(block)))
        yield block


def test_speech_detection_and_segment_saving(
    vad_streamer: SileroVADStreamer,
    mock_audio_data: torch.Tensor,
    output_dir: Path,
    mocker: MockerFixture,
) -> None:
    """Given one speech segment → Then one saved segment with correct timing"""
    for block in _split_into_blocks(mock_audio_data):
        indata = block.numpy().reshape(-1, 1)
        vad_streamer._audio_callback(indata, len(block), None, None)

    vad_streamer._signal_handler(None, None)

    segments = list(output_dir.glob("segment_*"))
    assert len(segments) == 1

    seg_dir = segments[0]
    wav_path = seg_dir / "sound.wav"
    json_path = seg_dir / "segment.json"

    assert wav_path.exists()
    assert json_path.exists()

    audio, sr = sf.read(str(wav_path))
    assert sr == 16000
    assert 1.9 <= len(audio) / sr <= 2.4

    with json_path.open() as f:
        meta = json.load(f)

    assert abs(meta["start_sec"] - 1.5) <= 0.15
    assert abs(meta["end_sec"] - 3.5) <= 0.15
    assert abs(meta["duration_sec"] - 2.0) <= 0.3


def test_no_speech_means_no_segments(output_dir: Path, mocker: MockerFixture) -> None:
    """Given only silence → Then zero segments saved"""
    mock_stream = mocker.MagicMock()
    mocker.patch("sounddevice.InputStream", return_value=mock_stream)

    streamer = SileroVADStreamer(
        threshold=0.9,
        output_dir=output_dir,
        save_segments=True,
        min_silence_duration_ms=100,
    )

    silence = torch.randn(16000 * 4) * 0.05
    for block in _split_into_blocks(silence):
        indata = block.numpy().reshape(-1, 1)
        streamer._audio_callback(indata, len(block), None, None)

    streamer._signal_handler(None, None)

    assert not any(output_dir.glob("segment_*"))


def test_multiple_speech_segments(output_dir: Path, mocker: MockerFixture) -> None:
    """Given two speech bursts → Then two segments saved"""
    mock_stream = mocker.MagicMock()
    mocker.patch("sounddevice.InputStream", return_value=mock_stream)

    streamer = SileroVADStreamer(
        output_dir=output_dir,
        save_segments=True,
        min_silence_duration_ms=800,
    )

    sr = 16000
    total_samples = sr * 12
    t = torch.linspace(0, 12, total_samples)

    audio = torch.zeros(total_samples)
    # Speech 1: 1.0–2.5s
    audio[int(1.0*sr):int(2.5*sr)] = 0.9 * torch.sin(2 * torch.pi * 800 * t[:int(1.5*sr)])
    # Speech 2: 6.0–8.0s
    audio[int(6.0*sr):int(8.0*sr)] = 0.9 * torch.sin(2 * torch.pi * 1200 * t[:int(2*sr)])

    for block in _split_into_blocks(audio):
        indata = block.numpy().reshape(-1, 1)
        streamer._audio_callback(indata, len(block), None, None)

    streamer._signal_handler(None, None)

    segments = sorted(output_dir.glob("segment_*"))
    assert len(segments) == 2
    assert (segments[0] / "sound.wav").exists()
    assert (segments[1] / "sound.wav").exists()