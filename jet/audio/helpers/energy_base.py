# jet_python_modules/jet/audio/utils.py   (new file or add to existing utils)

import numpy as np


def compute_amplitude(samples: np.ndarray) -> float:
    """Compute peak amplitude (max |x|).

    Range: 0.0 (true silence) → 1.0 (maximum possible loudness / 0 dBFS)
    Common values:
      - < 0.01   → very quiet / silence
      - 0.1–0.6  → normal speech
      - > 0.7    → loud speech
    """
    if len(samples) == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def compute_rms(samples: np.ndarray) -> float:
    """Root Mean Square – best simple measure of perceived loudness/energy.

    Range: 0.0 (true silence) → ~0.707 (full-scale sine wave)
    Typical speech values:
      - < 0.001     → silence / noise floor
      - 0.001–0.03  → very quiet / breath
      - 0.03–0.15   → normal conversational speech
      - 0.15–0.4+   → loud speech / shouting
    """
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def has_sound(samples: np.ndarray, threshold: float = 0.001) -> bool:
    """Return True if the audio contains meaningful sound.

    Now aligned with get_loudness_label():
      - rms < 0.001  → "silent"       → has_sound=False
      - rms >= 0.001 → "very_quiet" and above → has_sound=True
    """
    if len(samples) == 0:
        return False
    rms_value = compute_rms(samples)
    return rms_value >= threshold  # Note: >= so exactly 0.001 counts as sound


def rms_to_loudness_label(rms_value: float, silence_threshold: float = 0.001) -> str:
    """Return a human-readable loudness label based on RMS."""
    if rms_value < silence_threshold:
        return "silent"
    elif rms_value < 0.03:
        return "very_quiet"
    elif rms_value < 0.12:
        return "normal"
    elif rms_value < 0.25:
        return "loud"
    else:
        return "very_loud"
