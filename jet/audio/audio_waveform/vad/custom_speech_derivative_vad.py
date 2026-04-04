from typing import Dict, List, Optional, Tuple

import numpy as np


class DerivativeBasedVAD:
    """
    Final recommended Voice Activity Detection using derivatives.
    Good balance between stability on long speech and sensitivity to short bursts.
    """

    def __init__(
        self,
        activation_th: float = 0.45,
        deactivation_th: float = 0.40,
        min_speech_frames: int = 1,  # Allow single-frame peaks if raw is high
        min_silence_frames: int = 4,
        base_alpha: float = 0.50,  # Faster smoothing
        delta_weight: float = 0.96,
        onset_boost_base: float = 0.22,
        raw_peak_th: float = 0.78,
    ):
        self.activation_th = activation_th
        self.deactivation_th = deactivation_th
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames
        self.base_alpha = base_alpha
        self.delta_weight = delta_weight
        self.onset_boost_base = onset_boost_base
        self.raw_peak_th = raw_peak_th

    def _compute_delta(self, features: np.ndarray, order: int = 1) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        n = features.shape[1]
        delta = np.zeros_like(features, dtype=float)
        denom = 2 * (1 + 4)  # window = 2

        for t in range(n):
            num = 0.0
            for k in range(1, 3):
                tp = min(t + k, n - 1)
                tm = max(t - k, 0)
                num += k * (features[:, tp] - features[:, tm])
            delta[:, t] = num / denom

        if order == 2:
            return self._compute_delta(delta, order=1)
        return delta[0] if features.shape[0] == 1 else delta

    def process(
        self, speech_probs: np.ndarray, rms_energy: Optional[np.ndarray] = None
    ) -> Dict:
        probs = np.asarray(speech_probs, dtype=float)
        if len(probs) == 0:
            return {"speech_segments": [], "final_decisions": np.array([])}

        print("=== DerivativeBasedVAD Final Debug ===")
        smoothed, delta = self._adaptive_smooth(probs)
        decisions = np.zeros(len(probs), dtype=int)

        print("Frame | Raw    | Smoothed | ΔProb    | EffTh  | Boost   | Dec | Note")

        for t in range(len(probs)):
            eff_th = self.activation_th
            boost = 0.0
            note = ""

            if t > 0 and delta[t] > 0.05:
                boost = self.onset_boost_base * min(2.5, delta[t] / 0.10)
                eff_th = max(0.28, self.activation_th - boost)

            is_raw_peak = probs[t] >= self.raw_peak_th
            if smoothed[t] >= eff_th or is_raw_peak:
                decisions[t] = 1
                note = (
                    "RAW PEAK"
                    if is_raw_peak
                    else ("STRONG BOOST" if boost > 0.15 else "NORMAL")
                )

            print(
                f"{t:3d} | {probs[t]:.4f} | {smoothed[t]:.4f} | {delta[t]:+8.4f} | "
                f"{eff_th:.3f} | {boost:+6.3f} | {decisions[t]}   | {note}"
            )

        # Post-processing with tight control on short bursts
        final = decisions.copy()
        i = 0
        while i < len(final):
            if final[i] == 1:
                j = i
                while j < len(final) and final[j] == 1:
                    j += 1
                length = j - i

                if length < self.min_speech_frames:
                    final[i:j] = 0
                    print(f"→ Dropped too-short burst at {i}-{j - 1} ({length} frame)")
                elif length <= 5:  # short burst → limit hangover
                    final[j : j + 1] = 0  # allow at most 1 extra frame
                i = j
            else:
                i += 1

        # Extract segments
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
            alpha = np.clip(alpha, 0.04, 0.80)
            smoothed[t] = alpha * probs[t] + (1 - alpha) * smoothed[t - 1]
        return smoothed, delta


# ====================== Usage Example ======================
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

    vad = DerivativeBasedVAD()
    result = vad.process(speech_probs)
