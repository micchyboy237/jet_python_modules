def hybrid_strategy(
    probs: list[float],
    soft_limit: int = 450,
    hard_limit: int = 600,
    search_window: int = 100,
    speech_threshold: float = 0.5,
    valley_threshold: float = 0.6,
) -> list[int]:
    segments = []
    pos = 0
    n = len(probs)
    while pos < n:
        if probs[pos] <= speech_threshold:
            pos += 1
            continue
        speech_start = pos
        current_count = 0
        while pos < n and probs[pos] > speech_threshold:
            current_count += 1
            if current_count >= hard_limit:
                segments.append(current_count)
                pos += 1
                break
            if current_count > soft_limit:
                window_start = max(speech_start, pos - search_window + 1)
                recent_range = range(window_start, pos + 1)
                best_idx = min(recent_range, key=lambda j: (probs[j], -j))
                if probs[best_idx] < valley_threshold:
                    seg_len = best_idx - speech_start + 1
                    segments.append(seg_len)
                    speech_start = best_idx + 1
                    current_count = pos - best_idx
            pos += 1
        else:
            if current_count > 0:
                segments.append(current_count)
    return segments
