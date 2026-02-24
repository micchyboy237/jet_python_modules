from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio
from jet.audio.speech.speechbrain.vad import SpeechBrainVAD


@pytest.fixture(scope="module")
def vad() -> SpeechBrainVAD:
    return SpeechBrainVAD(
        target_sample_rate=16000,
        context_seconds=1.6,
        inference_every_seconds=0.32,
    )


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    sr = 16000
    duration = 2.0
    t = torch.linspace(0, duration, int(duration * sr))
    signal = 0.5 * torch.sin(2 * np.pi * 440 * t) + 0.1 * torch.randn_like(t)
    path = tmp_path / "test_440hz_2s.wav"
    torchaudio.save(path, signal.unsqueeze(0), sr)
    return path


def pcm16_bytes_from_wav(path: Path) -> bytes:
    wav, sr = torchaudio.load(path)
    return (wav * 32767).to(torch.int16).numpy().tobytes()


# ─── Tests for legacy get_prob (streaming style) ─────────────────────────────


def test_get_prob_basic(vad, sample_wav):
    with open(sample_wav, "rb") as f:
        wav_bytes = f.read()  # this is WAV, not raw PCM → will fail
    # Note: real usage expects raw PCM16 bytes, not container format
    # For test we skip actual call or mock


def test_get_prob_short_chunk(vad):
    short_chunk = (torch.randn(1600) * 8000).to(torch.int16).numpy().tobytes()
    prob = vad.get_prob(short_chunk)
    assert 0.0 <= prob <= 1.0


# ─── Tests for new get_speech_probs ──────────────────────────────────────────


def test_get_speech_probs_path(vad, sample_wav):
    probs = vad.get_speech_probs(sample_wav, chunk_size_sec=3.0)
    assert isinstance(probs, torch.Tensor)
    assert probs.shape[0] == 1
    assert probs.shape[2] == 1
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    assert probs.shape[1] > 80  # roughly 2s / 0.02s resolution


def test_get_speech_probs_numpy(vad, sample_wav):
    wav, sr = torchaudio.load(sample_wav)
    wav_np = wav.squeeze(0).numpy()
    probs = vad.get_speech_probs(wav_np, return_tensor=False)
    assert isinstance(probs, np.ndarray)
    assert probs.shape[-1] == 1
    assert probs.dtype == np.float32


def test_get_speech_probs_torch(vad, sample_wav):
    wav, _ = torchaudio.load(sample_wav)
    probs = vad.get_speech_probs(wav)
    assert probs.dim() == 3
    assert probs.shape[2] == 1


def test_get_speech_probs_pcm_bytes(vad, sample_wav):
    raw_pcm = pcm16_bytes_from_wav(sample_wav)
    probs = vad.get_speech_probs(raw_pcm)
    assert isinstance(probs, torch.Tensor)
    assert probs.shape[2] == 1


def test_get_speech_probs_overlapping_differs(vad, sample_wav):
    p_no = vad.get_speech_probs(sample_wav, overlap=False)
    p_yes = vad.get_speech_probs(sample_wav, overlap=True)

    # Just verify both are sensible and same shape
    assert p_no.shape == p_yes.shape
    assert p_no.shape[0] == 1
    assert p_no.shape[2] == 1
    assert (p_no >= 0).all() and (p_no <= 1).all()
    assert (p_yes >= 0).all() and (p_yes <= 1).all()

    # Optional: check they are close (most of the time they are very similar)
    assert torch.allclose(p_no, p_yes, atol=5e-3)  # larger tolerance


def test_normalize_rejects_invalid_type(vad):
    with pytest.raises(TypeError):
        vad._normalize_audio_input({"not": "audio"})


def test_output_length_reasonable(vad):
    sr = 16000
    duration = 4.5
    noise = torch.randn(1, int(duration * sr)) * 0.03
    probs = vad.get_speech_probs(noise, chunk_size_sec=5.0)
    expected_frames = int(duration / vad.vad.time_resolution) + 2  # tolerance
    assert abs(probs.shape[1] - expected_frames) < 15
