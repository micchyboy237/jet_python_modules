from typing import List, Union

import numpy as np


def quantize_np_to_int16(
    arr: np.ndarray, dither: bool = True, normalize: bool = True
) -> np.ndarray:
    """
    Quantize a numpy array of floats to int16 with proper handling.

    Args:
        arr: Input numpy array (typically in range [-1.0, 1.0])
        dither: Apply triangular dithering to reduce quantization distortion
        normalize: If True, normalize to full scale before quantizing

    Returns:
        Numpy array of dtype np.int16

    Raises:
        ValueError: If input contains NaN or inf values
    """
    # Ensure float64 for precision during processing
    audio = np.asarray(arr, dtype=np.float64)

    if not np.all(np.isfinite(audio)):
        raise ValueError("Input contains NaN or inf values")

    # Handle silence
    if np.max(np.abs(audio)) == 0:
        return np.zeros(audio.shape, dtype=np.int16)

    # Normalize if requested
    if normalize:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

    # Clip to valid range
    audio = np.clip(audio, -1.0, 1.0)

    # Scale to int16 range
    # Use 32768.0 for scaling, then clip to valid range
    # This ensures -1.0 maps to -32768 and +1.0 maps to +32767
    scaled = audio * 32768.0

    # Apply dithering before rounding
    if dither:
        # Triangular PDF dither (TPDF) - sum of two uniform RVs
        dither_noise = (
            np.random.uniform(-1, 1, audio.shape)
            + np.random.uniform(-1, 1, audio.shape)
        ) * 0.5  # 1 LSB peak-to-peak
        scaled += dither_noise

    # Round and clip to int16 range
    scaled = np.clip(np.round(scaled), -32768, 32767)

    return scaled.astype(np.int16)


def quantize_float_to_int16(
    float_list: List[float], dither: bool = True, normalize: bool = True
) -> List[int]:
    """
    Convert a list of floats to quantized int16 values.

    Args:
        float_list: List of float values
        dither: Apply dithering (recommended for audio)
        normalize: Normalize to full scale

    Returns:
        List of integers in range [-32768, 32767]
    """
    arr = np.asarray(float_list, dtype=np.float64)
    result = quantize_np_to_int16(arr, dither=dither, normalize=normalize)
    return result.tolist()


def dequantize_int16_to_float(
    quantized: Union[List[int], np.ndarray],
) -> Union[List[float], np.ndarray]:
    """
    Dequantize int16 values back to float in range [-1.0, 1.0].

    Uses symmetric scaling (division by 32768) to match quantization.
    """
    is_list = isinstance(quantized, list)

    # Convert to numpy array, preserving int16
    if is_list:
        arr = np.asarray(quantized, dtype=np.int16)
    else:
        arr = np.asarray(quantized).astype(np.int16)

    # Dequantize using 32768 to match quantization scaling
    # This ensures symmetric range: [-1.0, 0.999969...]
    float_data = arr.astype(np.float32) / 32768.0

    return float_data.tolist() if is_list else float_data


def quantize_to_int16(
    input_data: Union[List[float], np.ndarray],
    dither: bool = True,
    normalize: bool = True,
) -> Union[List[int], np.ndarray]:
    """
    Quantize float values to int16, accepting either list or numpy array.
    """
    arr = np.asarray(input_data, dtype=np.float64)

    if not np.all(np.isfinite(arr)):
        raise ValueError("Input contains NaN or inf values")

    result = quantize_np_to_int16(arr, dither=dither, normalize=normalize)

    return result.tolist() if isinstance(input_data, list) else result


if __name__ == "__main__":
    import argparse
    import shutil
    from pathlib import Path

    import librosa
    import numpy as np
    import soundfile as sf
    from jet.audio.audio_waveform.vad.vad_logging import linkify
    from rich.console import Console

    console = Console()

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Quantize and save audio.")
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    args = parser.parse_args()

    # Simplest usage - just use defaults
    audio, sr = librosa.load(args.audio_path, sr=None)
    quantized = quantize_np_to_int16(audio)

    output_path_wav = OUTPUT_DIR / "quantized.wav"

    sf.write(output_path_wav, quantized, sr)

    console.print(
        f"[bold green]Quantized audio saved to:[/bold green] {linkify(output_path_wav)}"
    )
