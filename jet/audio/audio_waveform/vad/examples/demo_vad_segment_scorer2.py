from jet.audio.audio_waveform.vad.vad_segment_scorer2 import (
    print_score,
    rank_candidates,
    score_segment_probs,
)

if __name__ == "__main__":
    # ── Example 1: basic scoring ──────────────────────────────────────────
    print("\n" + "═" * 44)
    print("  Example 1 – basic scoring")
    print("═" * 44)

    probs_good = [
        0.08,
        0.12,
        0.18,  # low baseline
        0.55,
        0.72,
        0.88,
        0.90,
        0.85,
        0.78,  # wide, high peak
        0.60,
        0.30,
        0.15,
        0.10,
        0.07,  # tail
    ]
    result = score_segment_probs(probs_good)
    print_score(result, "Well-shaped signal")

    # ── Example 2: narrow spike vs wide plateau ───────────────────────────
    print("\n" + "═" * 44)
    print("  Example 2 – narrow spike vs wide plateau")
    print("═" * 44)

    spike = [0.1] * 10 + [0.95] + [0.1] * 10  # single spike
    plateau = [0.1] * 3 + [0.70] * 15 + [0.1] * 3  # sustained plateau

    for label, probs in [("Narrow spike", spike), ("Wide plateau", plateau)]:
        r = score_segment_probs(probs, threshold=0.5)
        print_score(r, label)

    # ── Example 3: custom weights ─────────────────────────────────────────
    print("\n" + "═" * 44)
    print("  Example 3 – custom weights (peak-focused)")
    print("═" * 44)

    peak_weights = {
        "peak_area": 0.45,
        "peak_prominence": 0.35,
        "global_mean": 0.10,
        "global_median": 0.05,
        "coverage": 0.03,
        "stability": 0.02,
    }
    result_custom = score_segment_probs(probs_good, weights=peak_weights)
    print_score(result_custom, "Peak-focused weights")

    # ── Example 4: ranking multiple candidates ────────────────────────────
    print("\n" + "═" * 44)
    print("  Example 4 – ranking candidates")
    print("═" * 44)

    candidates = {
        "noisy": [0.3, 0.9, 0.2, 0.85, 0.1, 0.88, 0.15, 0.7],
        "consistent": [0.65, 0.70, 0.68, 0.72, 0.69, 0.71, 0.67, 0.70],
        "strong_peak": [0.1, 0.2, 0.85, 0.92, 0.91, 0.88, 0.2, 0.1],
        "weak": [0.2, 0.25, 0.22, 0.30, 0.28, 0.24, 0.21, 0.23],
    }

    ranking = rank_candidates(
        list(candidates.values()),
        labels=list(candidates.keys()),
        threshold=0.5,
    )

    print(f"\n  {'Rank':<6} {'Label':<14} {'Composite':>10}")
    print("  " + "─" * 32)
    for rank, (label, score, _) in enumerate(ranking, 1):
        print(f"  {rank:<6} {label:<14} {score:>10.4f}")

    # ── Example 5: access raw metrics programmatically ────────────────────
    print("\n" + "═" * 44)
    print("  Example 5 – programmatic access")
    print("═" * 44)

    r = score_segment_probs([0.1, 0.4, 0.8, 0.9, 0.85, 0.3, 0.1])
    print(f"\n  composite     : {r.composite}")
    print(f"  peak_height   : {r.peak_height}")
    print(f"  peak_width    : {r.peak_width}")
    print(f"  global_mean   : {r.global_mean}")
    print(f"  as dict keys  : {list(r.as_dict().keys())}")
