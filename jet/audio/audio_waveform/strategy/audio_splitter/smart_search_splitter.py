def smart_split_search_strategy(
    probs: list[float], max_speech_frame: int = 500, search_window: int = 100
) -> list[int]:
    # Assumes continuous speech block (all probs > 0.5)
    segments = []
    pos = 0
    n = len(probs)
    while pos < n:
        if pos + max_speech_frame > n:
            segments.append(n - pos)
            break
        limit_idx = pos + max_speech_frame - 1
        search_start = max(pos, limit_idx - search_window + 1)
        # Lowest prob; ties → latest frame (most natural)
        best_idx = min(range(search_start, limit_idx + 1), key=lambda j: (probs[j], -j))
        seg_len = best_idx - pos + 1
        segments.append(seg_len)
        pos = best_idx + 1
    return segments
