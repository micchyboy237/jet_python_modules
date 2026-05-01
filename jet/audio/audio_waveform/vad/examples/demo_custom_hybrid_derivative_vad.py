import numpy as np
from jet.audio.audio_waveform.vad.custom_hybrid_derivative_vad import DerivativeBasedVAD

if __name__ == "__main__":
    np.random.seed(42)

    speech_probs = np.concatenate(
        [
            np.full(10, 0.08),
            np.linspace(0.12, 0.92, 8),
            np.full(25, 0.94) + np.random.normal(0, 0.035, 25),
            np.linspace(0.90, 0.18, 9),
            np.full(15, 0.09) + np.random.normal(0, 0.02, 15),
            np.linspace(0.15, 0.85, 6),
            np.full(7, 0.07),
        ]
    )

    rms_energy = np.concatenate(
        [
            np.full(10, 0.012),
            np.linspace(0.015, 0.13, 8),
            np.full(25, 0.125) + np.random.normal(0, 0.01, 25),
            np.linspace(0.13, 0.022, 9),
            np.full(15, 0.014),
            np.linspace(0.018, 0.10, 6),
            np.full(7, 0.011),
        ]
    )

    print("\n=== Single sequence (ndarray) – original usage ===")
    vad = DerivativeBasedVAD(verbose=True, energy_weight=0.40)
    result = vad.process(speech_probs, rms_energy)

    # ── flat Python lists ──────────────────────────────────────────────
    print("\n=== Single sequence (plain Python list) ===")
    vad_quiet = DerivativeBasedVAD(verbose=False, energy_weight=0.40)
    result_list = vad_quiet.process(
        speech_probs.tolist(),  # flat list of floats
        rms_energy.tolist(),  # flat list of floats
    )
    print("Segments:", result_list["speech_segments"])

    # ── batch: list of lists ───────────────────────────────────────────
    print("\n=== Batch (list of lists) ===")
    batch_probs = [speech_probs.tolist(), speech_probs[:40].tolist()]
    batch_rms = [rms_energy.tolist(), rms_energy[:40].tolist()]
    results_batch = vad_quiet.process(batch_probs, batch_rms)
    for i, r in enumerate(results_batch):
        print(f"  Item {i}: segments={r['speech_segments']}")

    # ── batch: list of ndarrays ────────────────────────────────────────
    print("\n=== Batch (list of ndarrays) ===")
    results_arr = vad_quiet.process(
        [speech_probs, speech_probs[:40]],
        [rms_energy, rms_energy[:40]],
    )
    for i, r in enumerate(results_arr):
        print(f"  Item {i}: segments={r['speech_segments']}")

    # ── batch: 2-D ndarray ─────────────────────────────────────────────
    print("\n=== Batch (2-D ndarray, same-length sequences) ===")
    stacked_probs = np.stack([speech_probs, speech_probs])
    stacked_rms = np.stack([rms_energy, rms_energy])
    results_2d = vad_quiet.process(stacked_probs, stacked_rms)
    for i, r in enumerate(results_2d):
        print(f"  Item {i}: segments={r['speech_segments']}")
