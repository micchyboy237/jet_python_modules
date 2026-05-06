from typing import List, Union

import numpy as np


def quantize_np_to_int16(arr: np.ndarray) -> np.ndarray:
    """
    Quantize a numpy array of floats in range [-1.0, 1.0] to int16.

    Args:
        arr: Input numpy array with float values (should be in [-1.0, 1.0])

    Returns:
        Numpy array of dtype np.int16 with quantized values in [-32768, 32767]

    Raises:
        ValueError: If input contains NaN or inf values
        TypeError: If input is not a numpy array or convertible to one
    """
    # Ensure input is numpy array
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)

    # Handle NaN/inf
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input contains NaN or inf values")

    # Clip and quantize
    clipped: np.ndarray = np.clip(arr, -1.0, 1.0)
    int16_data: np.ndarray = (clipped * 32767).round().astype(np.int16)

    return int16_data


def quantize_float_to_int16(float_list: List[float]) -> List[int]:
    """
    Convert a list of floats to quantized int16 values.

    Args:
        float_list: List of float values in range [-1.0, 1.0]

    Returns:
        List of integers in range [-32768, 32767]

    Raises:
        ValueError: If input contains NaN or inf values
        TypeError: If input is not a list of numbers
    """
    # Convert to numpy array first
    arr: np.ndarray = np.asarray(float_list, dtype=np.float32)

    # Handle NaN/inf
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input contains NaN or inf values")

    # Clip and quantize
    clipped: np.ndarray = np.clip(arr, -1.0, 1.0)
    int16_data: np.ndarray = (clipped * 32767).round().astype(np.int16)

    return int16_data.tolist()


# Optional: Overloaded version that accepts both
def quantize_to_int16(
    input_data: Union[List[float], np.ndarray],
) -> Union[List[int], np.ndarray]:
    """
    Quantize float values to int16, accepting either list or numpy array.

    Args:
        input_data: Either a list of floats or a numpy array of floats

    Returns:
        If input is list: returns List[int]
        If input is numpy array: returns np.ndarray of dtype int16

    Raises:
        ValueError: If input contains NaN or inf values
        TypeError: If input type is not supported
    """
    arr: np.ndarray = np.asarray(input_data, dtype=np.float32)

    if not np.all(np.isfinite(arr)):
        raise ValueError("Input contains NaN or inf values")

    clipped: np.ndarray = np.clip(arr, -1.0, 1.0)
    result: np.ndarray = (clipped * 32767).round().astype(np.int16)

    if isinstance(input_data, list):
        return result.tolist()
    return result


def dequantize_int16_to_float(
    quantized: Union[List[int], np.ndarray],
) -> Union[List[float], np.ndarray]:
    """
    Dequantize int16 values back to float32 in range [-1.0, 1.0].

    Args:
        quantized: Either a list of ints or numpy array of int16 values
                  in range [-32768, 32767]

    Returns:
        Dequantized float values in range [-1.0, 1.0]

    Raises:
        ValueError: If input values are outside valid int16 range
    """
    # Convert to numpy array if needed
    if not isinstance(quantized, np.ndarray):
        arr = np.asarray(quantized, dtype=np.int16)
    else:
        arr = quantized.astype(np.int16)

    # Optional: Check range validity
    if np.any(arr < -32768) or np.any(arr > 32767):
        raise ValueError(
            f"Values outside int16 range: min={arr.min()}, max={arr.max()}"
        )

    # Dequantize back to float
    float_data = arr.astype(np.float32) / 32767.0

    # Return in same format as input
    if isinstance(quantized, list):
        return float_data.tolist()
    return float_data


# Example showing quantization and dequantization
if __name__ == "__main__":
    # Original floats
    original = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"Original: {original}")

    # Quantize
    quantized = quantize_np_to_int16(original)
    print(f"Quantized: {quantized}")

    # Dequantize
    recovered = dequantize_int16_to_float(quantized)
    print(f"Recovered: {recovered}")

    # Calculate quantization error
    error = np.abs(original - recovered)
    print(f"\nAbsolute error: {error}")
    print(f"Max error: {error.max():.6f}")
    print(f"Mean error: {error.mean():.6f}")

    # Demonstrate with list API
    print("\n--- List API example ---")
    float_list = [-1.0, -0.5, 0.0, 0.5, 1.0]
    quantized_list = quantize_float_to_int16(float_list)
    recovered_list = dequantize_int16_to_float(quantized_list)

    print(f"Original list: {float_list}")
    print(f"Quantized: {quantized_list}")
    print(f"Recovered: {recovered_list}")

    # Show the quantization levels
    print("\n--- Quantization resolution ---")
    print("Number of levels: 65536 (16-bit)")
    print(f"Step size: {2.0 / 65536:.8f}")
    print("Minimum representable difference: ~0.0000305")
