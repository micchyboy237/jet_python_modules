from typing import Union, Literal, Tuple
import numpy as np


def get_audio_dbfs(
    audio: np.ndarray,
    metric: Literal["peak", "true_peak_approx", "rms"] = "peak",
    return_type: Literal["float", "tuple"] = "float"
) -> Union[float, Tuple[float, float]]:
    """
    Calculate audio level in dBFS (decibels relative to full scale).

    Parameters
    ----------
    audio : np.ndarray
        Audio samples, typically float32/float64 ∈ [-1.0, 1.0]
    metric : {"peak", "true_peak_approx", "rms"}
        Which kind of level to return
    return_type : {"float", "tuple"}
        "float" → single value (peak by default)
        "tuple" → (peak_dbfs, rms_dbfs)

    Returns
    -------
    float or tuple[float, float]
        Level value(s) in dBFS
    """
    if audio.size == 0:
        if return_type == "tuple":
            return -np.inf, -np.inf
        return -np.inf

    audio = np.asarray(audio, dtype=np.float64)

    # True zero signal handling (most DAWs show -∞)
    if np.max(np.abs(audio)) == 0:
        if return_type == "tuple":
            return -np.inf, -np.inf
        return -np.inf

    peak = np.max(np.abs(audio))

    if metric == "rms":
        value = np.sqrt(np.mean(audio**2))
    elif metric == "true_peak_approx":
        # Very basic approximation - real TP needs interpolation + oversampling
        value = peak * 1.05  # rough, conservative estimate
    else:  # peak (default)
        value = peak

    # Protect very small values but still allow very negative dB
    value = max(value, 1e-12)

    dbfs = 20 * np.log10(value)

    if return_type == "tuple":
        rms = np.sqrt(np.mean(audio**2))
        rms = max(rms, 1e-12)
        rms_dbfs = 20 * np.log10(rms)
        return dbfs, rms_dbfs

    return dbfs


# ──────────────────────────────────────────────────────────────────────────────
#                           EXAMPLE USAGE
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simple demonstration
    import numpy as np
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    sine = 0.8 * np.sin(2 * np.pi * 1000 * t)

    print(f"Peak dBFS:      {get_audio_dbfs(sine):+.2f} dBFS")
    print(f"RMS dBFS:       {get_audio_dbfs(sine, metric='rms'):+.2f} dBFS")
    peak, rms = get_audio_dbfs(sine, return_type="tuple")
    print(f"Peak / RMS:     {peak:+.2f} / {rms:+.2f} dBFS")