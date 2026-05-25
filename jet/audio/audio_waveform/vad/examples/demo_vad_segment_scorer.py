from jet.audio.audio_waveform.vad.vad_segment_scorer import (
    score_balanced_speech,
    score_peak_confidence,
    score_sustained_speech,
)

if __name__ == "__main__":
    test_cases = {
        "narrow_high": [0.1, 0.1, 0.95, 0.1, 0.1],
        "wide_high": [0.9, 0.9, 0.9, 0.9, 0.1],
        "wide_moderate": [0.6, 0.6, 0.6, 0.6, 0.2],
    }

    print("Final VAD Scoring System")
    print("=" * 50)

    for name, probs in test_cases.items():
        balanced = score_balanced_speech(probs)
        sustained = score_sustained_speech(probs)
        peak = score_peak_confidence(probs)

        print(f"\n{name.upper()}: {probs}")
        print(f"  Balanced (general):     {balanced:.3f}")
        print(f"  Sustained (width-heavy): {sustained:.3f}")
        print(f"  Peak-focused:           {peak:.3f}")
