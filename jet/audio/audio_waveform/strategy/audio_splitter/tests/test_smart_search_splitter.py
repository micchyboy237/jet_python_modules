from jet.audio.audio_waveform.strategy.audio_splitter.smart_search_splitter import (
    smart_split_search_strategy,
)


def test_smart_split_search_flat():
    probs_flat = [0.9] * 600
    assert smart_split_search_strategy(probs_flat) == [500, 100]


def test_smart_split_search_dip():
    probs_dip = [0.9] * 400 + [0.3] + [0.9] * 200  # len=601
    result = smart_split_search_strategy(probs_dip)
    # Expected: split at the 0.3 dip
    assert result == [401, 200]
