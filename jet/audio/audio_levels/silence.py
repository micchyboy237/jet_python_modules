import numpy as np


def is_silent(
    audio: np.ndarray,
    threshold_db: float = -50.0,
    *,
    use_rms: bool = False,
    min_duration_sec: float = 0.0,
    sample_rate: int = 44100
) -> bool:
    if audio.size == 0:
        print("[DEBUG] Empty array → silent")
        return True

    audio = np.asarray(audio, dtype=np.float64).flatten()  # force mono

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio / np.iinfo(audio.dtype).max

    audio -= np.mean(audio)  # very basic DC removal

    # Calculate energy
    if use_rms:
        energy = np.sqrt(np.mean(audio**2))
    else:
        energy = np.max(np.abs(audio))

    # Minimum duration check FIRST, even for zeros
    if min_duration_sec > 0:
        min_samples = int(min_duration_sec * sample_rate)
        if len(audio) < min_samples:
            print("[DEBUG] Too short - not enough samples for min duration")
            return False

    if energy == 0:
        print("[DEBUG] Energy exactly 0 → silent")
        return True

    energy_db = 20 * np.log10(energy + 1e-12)
    
    print(f"[DEBUG] energy_db = {energy_db:.2f} dBFS  |  threshold = {threshold_db:.1f} dBFS")
    print(f"[DEBUG] max_peak_linear = {np.max(np.abs(audio)):.8f}")

    is_quiet_enough = energy_db <= threshold_db

    print(f"[DEBUG] is_quiet_enough = {is_quiet_enough}")

    return is_quiet_enough


# ──────────────────────────────────────────────────────────────────────────────
#                          USAGE EXAMPLES
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example 1: Complete silence
    silent = np.zeros(22050)  # 0.5 second silence @ 44.1kHz
    print("Complete silence (peak):   ", is_silent(silent))                    # True
    print("Complete silence (RMS):    ", is_silent(silent, use_rms=True))      # True

    # Example 2: Normal speaking level
    t = np.linspace(0, 0.5, 22050)
    speech = 0.4 * np.sin(2 * np.pi * 220 * t)  # ~ -8 dBFS peak
    print("Loud speech (peak):        ", is_silent(speech))                    # False
    print("Loud speech (-60dB thresh):", is_silent(speech, threshold_db=-60))  # False

    # Example 3: Very quiet background noise (~ -55 dBFS RMS)
    noise = np.random.normal(0, 0.0018, 22050)  # ≈ -55 dBFS RMS
    print("Quiet noise (peak, -50dB): ", is_silent(noise))                     # True
    print("Quiet noise (RMS, -50dB):  ", is_silent(noise, use_rms=True))       # False (RMS higher than peak)

    # Example 4: Requiring minimum silence duration
    short_silence = np.zeros(1000)  # ~23ms @ 44.1kHz
    print("Short silence (min 0.1s):  ", is_silent(
        short_silence,
        min_duration_sec=0.1,
        sample_rate=44100
    ))  # False

    long_silence = np.zeros(8820)  # ~0.2s
    print("Long silence (min 0.1s):   ", is_silent(
        long_silence,
        min_duration_sec=0.1,
        sample_rate=44100
    ))  # True