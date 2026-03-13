from jet.audio.audio_waveform.strategy.audio_splitter.soft_max_splitter import (
    soft_max_strategy,
)


def test_soft_max_strategy_pure():
    probs_pure = [0.9] * 1600
    assert soft_max_strategy(probs_pure, 450, 600) == [600, 600, 400]


def test_soft_max_strategy_silence():
    probs_silence = [0.9] * 500 + [0.1] + [0.9] * 100
    result = soft_max_strategy(probs_silence)
    # Silence at 500 (> soft) → clean split
    assert result == [500, 100]
