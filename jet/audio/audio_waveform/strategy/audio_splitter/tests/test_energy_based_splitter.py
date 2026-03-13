from jet.audio.audio_waveform.strategy.audio_splitter.energy_based_splitter import (
    energy_based_strategy,
)


def test_energy_based_strategy_valley():
    energies = [1.0] * 400 + [0.1] + [1.0] * 200
    result = energy_based_strategy(energies)
    # Expected: split at energy valley
    assert result == [401, 200]
