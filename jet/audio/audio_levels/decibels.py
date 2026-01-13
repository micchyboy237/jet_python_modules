# audio_decibels.py

from typing import Union, Tuple, Literal
import numpy as np


def get_audio_decibels(
    audio: np.ndarray,
    reference: Literal["full_scale", "rms", "peak", "custom"] = "full_scale",
    custom_ref: float = 1.0,
    db_type: Literal["db", "dbfs", "dbtp"] = "dbfs",
    return_type: Literal["float", "tuple"] = "float"
) -> Union[float, Tuple[float, float]]:
    """
    Calculate decibel level of audio signal with flexible reference points.

    Parameters:
    -----------
    audio : np.ndarray
        Audio samples (float32/float64, typically -1.0 to 1.0 range)
        Can be 1D (mono) or 2D (multi-channel)

    reference : str
        'full_scale'    → 0 dBFS = digital full scale (most common in DAWs)
        'rms'           → 0 dB = RMS of full scale sine wave (~-3 dBFS)
        'peak'          → 0 dB = absolute peak value
        'custom'        → use custom_ref value as 0 dB reference

    custom_ref : float
        Reference value when reference="custom"

    db_type : str
        'dbfs'  → digital full scale (most common)
        'db'    → plain decibels (same math, different name)
        'dbtp'  → dB True Peak (requires extra headroom consideration)

    return_type : str
        'float'  → single value (mono or average of channels)
        'tuple'  → (peak_db, rms_db) useful for metering

    Returns:
    --------
    float or tuple[float, float]
        Decibel value(s) relative to chosen reference
    """
    if audio.size == 0:
        return -np.inf if return_type == "float" else (-np.inf, -np.inf)

    # Make sure we work with float
    audio = audio.astype(np.float64)

    # Compute basic statistics
    peak_abs = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))

    # Handle true silence case
    if peak_abs == 0:
        if return_type == "tuple":
            return -np.inf, -np.inf
        return -np.inf

    if reference == "full_scale":
        ref_value = 1.0
    elif reference == "rms":
        ref_value = 1.0 / np.sqrt(2)  # ≈ 0.707 → 0 dB RMS = -3.01 dBFS
    elif reference == "peak":
        ref_value = peak_abs if peak_abs > 0 else 1.0
    elif reference == "custom":
        ref_value = abs(custom_ref) if custom_ref != 0 else 1.0
    else:
        raise ValueError(f"Unknown reference: {reference}")

    # Avoid log(0) but allow true silence handling above
    peak_abs = max(peak_abs, 1e-38)   # very small but not zero → avoids -inf
    rms = max(rms, 1e-38)

    # Calculate dB
    peak_db = 20 * np.log10(peak_abs / ref_value)
    rms_db = 20 * np.log10(rms / ref_value)

    if db_type == "dbtp":
        # Very rough true-peak approximation (real TP needs interpolation + oversampling)
        peak_db += 0.0  # placeholder - real dBTP usually needs more work

    if return_type == "tuple":
        return peak_db, rms_db
    else:
        # Most people want peak when asking for "the level"
        return peak_db


# ──────────────────────────────────────────────────────────────────────────────
#                          USAGE EXAMPLES
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Fake audio signals (replace with real audio)
    sr = 44100
    t = np.linspace(0, 1, sr)

    # 1. Full scale sine wave → should be ≈ -3.01 dBFS RMS, 0 dBFS peak
    sine_full = 0.999 * np.sin(2 * np.pi * 440 * t)

    print("Full scale sine:")
    print("  Peak dBFS  :", get_audio_decibels(sine_full, "full_scale"))
    print("  RMS dBFS   :", get_audio_decibels(sine_full, "full_scale", return_type="tuple")[1])
    print()

    # 2. -12 dBFS sine
    sine_minus12 = sine_full * 10**(-12/20)

    print("-12 dBFS sine:")
    print("  Peak  :", round(get_audio_decibels(sine_minus12), 2), "dBFS")
    print("  RMS   :", round(get_audio_decibels(sine_minus12, reference="rms"), 2), "dB RMS")
    print()

    # 3. Very quiet signal
    quiet = sine_full * 0.0001  # -80 dBFS-ish
    print("Quiet signal peak:", get_audio_decibels(quiet), "dBFS")
    print()

    # 4. Stereo example with tuple return
    stereo = np.vstack([sine_full * 0.7, sine_full * 0.4])
    peak_l, rms_l = get_audio_decibels(stereo[0], return_type="tuple")
    peak_r, rms_r = get_audio_decibels(stereo[1], return_type="tuple")

    print("Stereo example:")
    print(f"  L : peak {peak_l:5.2f} dBFS   RMS {rms_l:5.2f} dBFS")
    print(f"  R : peak {peak_r:5.2f} dBFS   RMS {rms_r:5.2f} dBFS")