def soft_max_strategy(
    probs: list[float],
    soft_limit: int = 450,
    hard_limit: int = 600,
    speech_threshold: float = 0.5,
) -> list[int]:
    segments = []
    count = 0
    for p in probs:
        if p > speech_threshold:
            count += 1
            if count >= hard_limit:
                segments.append(count)
                count = 0
        else:
            if count > 0:
                segments.append(count)  # split on silence (if > soft or short)
                count = 0
    if count > 0:
        segments.append(count)
    return segments
