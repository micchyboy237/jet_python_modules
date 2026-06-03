import numpy as np
from jet.audio.normalization.quant2 import (
    dequantize_int16_to_float,
    quantize_np_to_int16,
)

# Example showing quantization and dequantization
if __name__ == "__main__":
    print("=== Testing quantization symmetry ===")

    test_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    # Original approach (multiply by 32767)
    original_method = (np.clip(test_values, -1, 1) * 32767).round().astype(np.int16)
    print(f"Original method: {original_method}")
    print(f"  Min: {original_method.min()}, Max: {original_method.max()}")
    print(f"  Note: -1.0 maps to {original_method[0]}, not -32768!")

    # Corrected approach
    corrected = quantize_np_to_int16(test_values, dither=False, normalize=False)
    print(f"\nCorrected method: {corrected}")
    print(f"  Min: {corrected.min()}, Max: {corrected.max()}")

    # Round-trip test
    recovered = dequantize_int16_to_float(corrected)
    print(f"\nRound-trip: {recovered}")
    print(f"Max error: {np.max(np.abs(test_values - recovered)):.8f}")

    print("\n=== Dithering demonstration ===")
    # Small signal without dither
    small = np.sin(2 * np.pi * 1000 * np.linspace(0, 1, 1000)) * 0.0001
    without_dither = quantize_np_to_int16(small, dither=False, normalize=False)
    with_dither = quantize_np_to_int16(small, dither=True, normalize=False)

    unique_vals_no_dither = len(np.unique(without_dither))
    unique_vals_dithered = len(np.unique(with_dither))
    print("Small signal (-80dB):")
    print(f"  Without dither: {unique_vals_no_dither} unique values (harsh distortion)")
    print(f"  With dither: {unique_vals_dithered} unique values (smooth noise floor)")
