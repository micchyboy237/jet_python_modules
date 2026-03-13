from jet.audio.audio_waveform.strategy.audio_splitter.hybrid_splitter import (
    hybrid_strategy,
)


def test_hybrid_strategy_pure():
    probs_pure = [0.9] * 1600
    assert hybrid_strategy(probs_pure) == [600, 600, 400]


def test_hybrid_strategy_valley():
    probs_valley = [0.9] * 460 + [0.55] + [0.9] * 200
    assert hybrid_strategy(probs_valley) == [461, 200]


def test_hybrid_strategy_silence():
    probs_sil = [0.9] * 500 + [0.4] + [0.9] * 100
    assert hybrid_strategy(probs_sil) == [500, 100]
