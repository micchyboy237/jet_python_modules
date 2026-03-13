from jet.audio.audio_waveform.strategy.audio_splitter.probability_valley_splitter import (
    probability_valley_strategy,
)


def test_probability_valley_strategy_valley():
    probs_dip = [0.9] * 400 + [0.3] + [0.9] * 200
    result = probability_valley_strategy(probs_dip)
    assert result == [401, 200]
