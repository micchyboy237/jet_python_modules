# tests/test_speechbrain_vad.py
import numpy as np
import pytest
from jet.audio.speech.speechbrain.vad import SpeechBrainVAD


@pytest.fixture(scope="module")
def vad():
    """Shared VAD instance — loads model only once per module"""
    return SpeechBrainVAD()


def test_empty_chunk_returns_empty_list(vad):
    # Given a fresh VAD instance and an empty chunk
    chunk = np.array([], dtype=np.float32)

    # When we process it
    probs = vad.get_speech_probs(chunk)

    # Then we get an empty list
    expected = []
    assert probs == expected, "Empty chunk should return empty list"


def test_small_initial_chunk_returns_growing_probs_list():
    # Given a fresh VAD (reset by new instance or assume fresh)
    vad = SpeechBrainVAD()  # fresh to control state

    # When we feed very small chunks step by step
    chunk1 = np.zeros(1600, dtype=np.float32)  # 0.1 s silence
    probs1 = vad.get_speech_probs(chunk1)

    chunk2 = np.zeros(3200, dtype=np.float32)  # +0.2 s
    probs2 = vad.get_speech_probs(chunk2)

    # Then list length increases until context is filled
    expected_len1 = 10  # roughly 0.1 s / 0.01 s = 10 frames
    expected_len2 = 30  # cumulative ~0.3 s → ~30 frames

    assert len(probs1) == pytest.approx(expected_len1, abs=3), (
        "Should return ~10 frames after 0.1 s"
    )
    assert len(probs2) > len(probs1), "List should grow with more audio"
    assert len(probs2) <= 55, "Should not exceed context frame count yet"


def test_after_full_context_length_is_stable(vad):
    # Given a VAD instance
    context_samples = vad.context_samples  # 8000
    frame_count_approx = 50

    # When we feed enough audio to fill + overflow context
    silence = np.zeros(context_samples // 2, dtype=np.float32)
    vad.get_speech_probs(silence)  # half
    vad.get_speech_probs(silence)  # full
    vad.get_speech_probs(np.zeros(4000, dtype=np.float32))  # overflow

    probs = vad.get_speech_probs(np.zeros(100, dtype=np.float32))

    # Then returned list length stabilizes at ~50 frames
    assert len(probs) == pytest.approx(frame_count_approx, abs=5)


def test_ring_buffer_wrap_keeps_newest_content():
    # Given fresh VAD
    vad = SpeechBrainVAD()

    # When we fill with silence then add loud tone at the end
    silence = np.zeros(8000, dtype=np.float32)
    vad.get_speech_probs(silence)  # fill with silence

    # Create speech-like signal (sine wave)
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    speech = 0.8 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    probs_after_speech = vad.get_speech_probs(speech)

    # Then most of the probabilities should be high (speech moved to newest part)
    expected_high = [p for p in probs_after_speech if p > 0.7]
    assert len(expected_high) > 25, "Newest half should show high speech probability"


def test_prob_values_are_in_valid_range(vad):
    # Given any processing
    chunk = np.random.randn(3200).astype(np.float32)  # noise
    probs = vad.get_speech_probs(chunk)

    # Then all values are floats between 0 and 1
    assert all(isinstance(p, float) for p in probs)
    assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities must be in [0, 1]"


def test_continuous_silence_gives_low_probs():
    # Given fresh VAD filled with silence
    vad = SpeechBrainVAD()
    silence = np.zeros(12000, dtype=np.float32)  # > context
    probs = vad.get_speech_probs(silence)

    # Then most probs should be low
    high_probs = [p for p in probs if p > 0.4]
    assert len(high_probs) < 10, "Silence should produce mostly low probabilities"


@pytest.mark.xfail(reason="Exact values depend on model; qualitative only")
def test_speech_like_signal_gives_high_probs():
    # Given fresh VAD
    vad = SpeechBrainVAD()

    # 440 Hz sine — speech-like energy
    sr = 16000
    t = np.linspace(0, 0.6, int(0.6 * sr), endpoint=False)
    speech = 0.7 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    probs = vad.get_speech_probs(speech)

    # Then average prob should be reasonably high
    mean_prob = np.mean(probs)
    assert mean_prob > 0.6, f"Expected higher speech prob, got mean={mean_prob:.3f}"
