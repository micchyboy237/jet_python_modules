from jet.audio.audio_waveform.strategy.audio_splitter.hard_splitter import (
    hard_split_strategy,
)


def test_hard_split_strategy():
    probs = [0.9] * 1600
    result = hard_split_strategy(probs)
    # Expected: [500, 500, 500, 100]
    assert result == [500, 500, 500, 100]
