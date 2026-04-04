from typing import Dict, List, Optional, Tuple

import numpy as np


class DerivativeBasedVAD:
    """
    Final corrected version with proper decision flow + rich debug logging.
    """

    def __init__(
        self,
        activation_th: float = 0.45,
        min_speech_frames: int = 1,
        base_alpha: float = 0.48,
        delta_weight: float = 0.97,
        onset_boost_base: float = 0.24,
        raw_peak_th: float = 0.78,
        use_energy_gating: bool = False,
        min_rms_for_speech: float = 0.015,
    ):
        self.activation_th = activation_th
        self.min_speech_frames = min_speech_frames
        self.base_alpha = base_alpha
        self.delta_weight = delta_weight
        self.onset_boost_base = onset_boost_base
        self.raw_peak_th = raw_peak_th
        self.use_energy_gating = use_energy_gating
        self.min_rms_for_speech = min_rms_for_speech

    def _compute_delta(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        n = features.shape[1]
        delta = np.zeros_like(features, dtype=float)
        denom = 2 * (1 + 4)

        for t in range(n):
            num = 0.0
            for k in range(1, 3):
                tp = min(t + k, n - 1)
                tm = max(t - k, 0)
                num += k * (features[:, tp] - features[:, tm])
            delta[:, t] = num / denom
        return delta[0] if features.shape[0] == 1 else delta

    def process(
        self, speech_probs: np.ndarray, rms_energy: Optional[np.ndarray] = None
    ) -> Dict:
        probs = np.asarray(speech_probs, dtype=float)
        rms = np.asarray(rms_energy, dtype=float) if rms_energy is not None else None

        print("=== DerivativeBasedVAD Detailed Debug ===")
        smoothed, delta = self._adaptive_smooth(probs)
        initial_decisions = np.zeros(len(probs), dtype=int)

        print("\n--- Per-Frame Decision Log ---")
        print(
            "Frame | RawProb | Smoothed | ΔProb    | EffTh  | Boost   | RMS     | InitDec | Reason"
        )

        for t in range(len(probs)):
            eff_th = self.activation_th
            boost = 0.0
            reason = ""

            if t > 0 and delta[t] > 0.05:
                boost = self.onset_boost_base * min(2.8, delta[t] / 0.10)
                eff_th = max(0.26, self.activation_th - boost)

            is_raw_peak = probs[t] >= self.raw_peak_th
            init_dec = 1 if (smoothed[t] >= eff_th or is_raw_peak) else 0

            if self.use_energy_gating and rms is not None and init_dec == 1:
                if rms[t] < self.min_rms_for_speech:
                    init_dec = 0
                    reason = f"ENERGY REJECT (rms={rms[t]:.4f})"
                else:
                    reason = "ENERGY OK"

            if init_dec == 1 and not reason:
                reason = (
                    "RAW PEAK"
                    if is_raw_peak
                    else ("BOOST" if boost > 0.15 else "SMOOTHED OK")
                )

            rms_str = f"{rms[t]:.4f}" if rms is not None else "N/A"
            print(
                f"{t:3d} | {probs[t]:.4f} | {smoothed[t]:.4f} | {delta[t]:+8.4f} | "
                f"{eff_th:.3f} | {boost:+6.3f} | {rms_str} | {init_dec}      | {reason}"
            )

            initial_decisions[t] = init_dec

        # === Post-processing ===
        final = initial_decisions.copy()
        print("\n--- Post-Processing Log ---")

        i = 0
        while i < len(final):
            if final[i] == 1:
                j = i
                while j < len(final) and final[j] == 1:
                    j += 1
                length = j - i
                print(f"Found speech candidate {i}–{j - 1} (length = {length} frames)")

                if length < self.min_speech_frames:
                    final[i:j] = 0
                    print(f"  → DROPPED: too short (< {self.min_speech_frames} frames)")
                elif length <= 6:
                    final[j : j + 1] = 0
                    print("  → Limited hangover for short burst")
                i = j
            else:
                i += 1

        # Extract final segments
        segments: List[Tuple[int, int]] = []
        i = 0
        while i < len(final):
            if final[i] == 1:
                start = i
                while i < len(final) and final[i] == 1:
                    i += 1
                segments.append((start, i))
            else:
                i += 1

        print("\n=== FINAL RESULT ===")
        print("Speech segments:", segments)
        print("Total speech frames:", final.sum())
        print(f"Max smoothed: {np.max(smoothed):.4f}")

        if not segments:
            print("WARNING: All candidates were dropped in post-processing.")

        return {
            "speech_segments": segments,
            "final_decisions": final,
            "smoothed_probs": smoothed,
            "delta_probs": delta,
            "original_probs": probs,
        }

    def _adaptive_smooth(self, probs: np.ndarray):
        smoothed = np.zeros_like(probs)
        smoothed[0] = probs[0]
        delta = self._compute_delta(probs)

        change_norm = np.abs(delta) / (np.max(np.abs(delta)) + 1e-8)

        for t in range(1, len(probs)):
            alpha = self.base_alpha - self.delta_weight * change_norm[t - 1]
            alpha = np.clip(alpha, 0.03, 0.78)
            smoothed[t] = alpha * probs[t] + (1 - alpha) * smoothed[t - 1]
        return smoothed, delta


# ====================== Test ======================
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

    vad = DerivativeBasedVAD(use_energy_gating=False)
    result = vad.process(speech_probs, rms_energy)
