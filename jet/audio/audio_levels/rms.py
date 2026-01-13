# audio_rms.py

from typing import Union, Literal
import numpy as np


def get_audio_rms(
    audio: np.ndarray,
    reference: Literal["full_scale", "custom"] = "full_scale",
    custom_ref: float = 1.0,
    return_db: bool = True
) -> Union[float, np.ndarray]:
    """
    Calculate RMS level of audio signal (in linear or dB scale).

    Parameters:
    -----------
    audio : np.ndarray
        Audio samples (typically float32/float64, -1.0 to 1.0 range)
        Supports mono (1D) and multi-channel (2D)

    reference : {'full_scale', 'custom'}
        'full_scale' → reference = 1.0 (standard digital full scale)
        'custom'     → use custom_ref as reference value

    custom_ref : float
        Reference amplitude when reference="custom"

    return_db : bool
        True  → return in decibels (dBFS or dB relative to custom ref)
        False → return linear RMS value

    Returns:
    --------
    float or np.ndarray
        - Single float for mono or when averaging channels
        - Per-channel RMS when input is 2D and return_db=False
        - dB values are relative to chosen reference
    """
    if audio.size == 0:
        return -np.inf if return_db else 0.0

    audio = np.asarray(audio, dtype=np.float64)

    # Compute mean square (power) first - avoids sqrt(0) issues
    if audio.ndim == 1:
        power = np.mean(audio ** 2)
    else:
        power = np.mean(audio ** 2, axis=-1)

    # Explicit true silence handling - most DAWs/meters show -∞
    if np.all(power == 0):
        if return_db:
            if isinstance(power, np.ndarray) and power.size > 1:
                return np.full_like(power, -np.inf, dtype=float)
            return -np.inf
        else:
            return np.zeros_like(power) if getattr(power, "ndim", 0) > 0 else 0.0

    # Normal path - now safe
    rms = np.sqrt(power)

    # Apply reference
    if reference == "full_scale":
        ref_value = 1.0
    elif reference == "custom":
        ref_value = float(abs(custom_ref)) if custom_ref != 0 else 1.0
    else:
        raise ValueError(f"Unsupported reference: {reference}")

    if return_db:
        db_value = 20 * np.log10(rms / ref_value)
        if isinstance(db_value, np.ndarray) and db_value.size == 1:
            return float(db_value)
        return db_value

    return rms


# ──────────────────────────────────────────────────────────────────────────────
#                          USAGE EXAMPLES
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example signals
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)

    full_sine = np.sin(2 * np.pi * 440 * t)              # peak ≈ 1.0
    minus_12db = full_sine * 10**(-12/20)                # peak ≈ -12 dBFS
    stereo = np.vstack([full_sine * 0.8, full_sine * 0.5])

    print("Full scale sine:")
    print(f"  RMS dBFS: {get_audio_rms(full_sine):+.2f} dB")
    print(f"  RMS linear: {get_audio_rms(full_sine, return_db=False):.6f}")

    print("\n-12 dBFS sine:")
    print(f"  RMS: {get_audio_rms(minus_12db):+.2f} dBFS")

    print("\nStereo example:")
    rms_stereo_db = get_audio_rms(stereo)
    print(f"  Left:  {rms_stereo_db[0]:+.2f} dBFS")
    print(f"  Right: {rms_stereo_db[1]:+.2f} dBFS")