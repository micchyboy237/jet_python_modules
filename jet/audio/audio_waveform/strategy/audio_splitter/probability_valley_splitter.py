# Reuses the smart search (identical valley logic)
from jet.audio.audio_waveform.strategy.audio_splitter.smart_search_splitter import (
    smart_split_search_strategy,
)


def probability_valley_strategy(
    probs: list[float], max_speech_frame: int = 500, search_window: int = 100
) -> list[int]:
    return smart_split_search_strategy(probs, max_speech_frame, search_window)
