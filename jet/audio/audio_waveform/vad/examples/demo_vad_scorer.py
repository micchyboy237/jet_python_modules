"""
demo_vad_scorer.py
==================
Demonstrates VADScorer quality cases AND the frame-rate invariance fix.

Run:
    python demo_vad_scorer.py
"""

from jet.audio.audio_waveform.vad.vad_scorer import VADScorer


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_cases(cases: dict, frame_shift_ms: float = 10.0) -> None:
    fs = frame_shift_ms / 1000.0
    for name, probs in cases.items():
        print(f"\n{'─' * 60}")
        print(f"  {name}")
        print(VADScorer(probs, frame_shift_s=fs).report())


# ---------------------------------------------------------------------------
# Standard quality cases
# ---------------------------------------------------------------------------

def demo_quality_cases() -> None:
    print("\n" + "═" * 60)
    print("  QUALITY CASES  (10 ms frame shift)")
    print("═" * 60)

    real_segment = [
        0.156, 0.163, 0.158, 0.185, 0.237, 0.276, 0.305, 0.343, 0.339, 0.286,
        0.258, 0.227, 0.174, 0.15,  0.136, 0.135, 0.15,  0.169, 0.212, 0.304,
        0.335, 0.361, 0.387, 0.423, 0.384, 0.414, 0.421, 0.399, 0.379, 0.375,
        0.349, 0.348, 0.378, 0.391, 0.418, 0.423, 0.431, 0.433, 0.436, 0.415,
        0.415, 0.397, 0.402, 0.386, 0.401, 0.407, 0.422, 0.408, 0.388, 0.308,
        0.239, 0.168, 0.095, 0.045, 0.042, 0.032, 0.024, 0.02,  0.015, 0.011,
        0.008, 0.005, 0.004, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001,
    ] + [0.0] * 60

    cases = {
        "Your real segment (max=0.436, never crosses tau)": real_segment,
        "Very good — crisp bimodal speech": [
            0.01, 0.02, 0.01, 0.95, 0.97, 0.98, 0.96, 0.99,
            0.97, 0.02, 0.01, 0.98, 0.97, 0.96, 0.02, 0.01,
        ],
        "Bad — all probs stuck near 0.5": [
            0.52, 0.48, 0.55, 0.51, 0.49, 0.53, 0.47, 0.50,
            0.52, 0.48, 0.55, 0.51, 0.49, 0.53, 0.47, 0.50,
        ],
        "Fair — mixed signal": [
            0.05, 0.60, 0.75, 0.50, 0.88, 0.45, 0.92, 0.30,
            0.55, 0.10, 0.70, 0.48, 0.85, 0.20, 0.65, 0.40,
        ],
    }

    _run_cases(cases, frame_shift_ms=10.0)


# ---------------------------------------------------------------------------
# Frame-rate invariance demo
# ---------------------------------------------------------------------------

def demo_invariance() -> None:
    """
    Same audio, two VAD frame shifts.  smoothness_score and segment lengths
    should be identical (or very close) after the fix.
    """
    print("\n\n" + "═" * 60)
    print("  FRAME-RATE INVARIANCE DEMO")
    print("  Same audio content, two different frame shifts.")
    print("  smoothness_score and segment lengths should match.")
    print("═" * 60)

    # Represent the same ~300 ms speech burst + silence
    # At 10 ms/frame: 30 speech frames + 20 silence frames
    probs_10ms = [0.95] * 30 + [0.02] * 20

    # At 25 ms/frame: 12 speech frames + 8 silence frames (~same duration)
    probs_25ms = [0.95] * 12 + [0.02] * 8

    shifts = [
        ("10 ms shift (100 fps) — FireRedVAD default", probs_10ms, 0.010),
        ("25 ms shift  (40 fps) — coarser VAD",        probs_25ms, 0.025),
    ]

    rows = []
    for label, probs, fs in shifts:
        m = VADScorer(probs, frame_shift_s=fs).metrics()
        rows.append((label, m))
        print(f"\n  ▶ {label}")
        print(f"    jitter (raw)         : {m.jitter:.4f}  ← differs by frame rate (expected)")
        print(f"    jitter_per_s         : {m.jitter_per_s:.4f}  ← should be ~same ✓")
        print(f"    smoothness_score     : {m.smoothness_score:.4f}  ← should be ~same ✓")
        print(f"    mean speech seg (fr) : {m.mean_speech_segment_len:.1f} frames")
        print(f"    mean speech seg (s)  : {m.mean_speech_segment_len_s*1000:.0f} ms  ← should be ~same ✓")
        print(f"    composite_score      : {m.composite_score:.4f}")

    # Validation
    m0, m1 = rows[0][1], rows[1][1]
    tol = 0.05
    print("\n  ── Validation ──────────────────────────────────────")
    checks = [
        ("jitter_per_s",            m0.jitter_per_s,            m1.jitter_per_s),
        ("smoothness_score",        m0.smoothness_score,        m1.smoothness_score),
        ("mean_speech_seg_s (ms)",  m0.mean_speech_segment_len_s * 1000,
                                    m1.mean_speech_segment_len_s * 1000),
    ]
    all_ok = True
    for name, v0, v1 in checks:
        ok = abs(v0 - v1) <= tol * max(abs(v0), abs(v1), 1e-9) * 100
        status = "✓ OK" if ok else "✗ FAIL"
        if not ok:
            all_ok = False
        print(f"    {name:<30} {v0:8.3f}  vs  {v1:8.3f}   {status}")
    print()
    print("  Result:", "✓ All invariance checks passed." if all_ok else "✗ Some checks failed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_quality_cases()
    demo_invariance()
