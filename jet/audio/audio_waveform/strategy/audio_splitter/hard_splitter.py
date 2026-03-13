def hard_split_strategy(
    probs: list[float], max_speech_frame: int = 500, speech_threshold: float = 0.5
) -> list[int]:
    segments = []
    count = 0
    for p in probs:
        if p > speech_threshold:
            count += 1
            if count >= max_speech_frame:
                segments.append(count)
                count = 0
        else:
            if count > 0:
                segments.append(count)
                count = 0
    if count > 0:
        segments.append(count)
    return segments
