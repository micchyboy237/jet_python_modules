"""Loudness and peak normalization utilities for audio signals.

Provides functions for peak normalization, loudness normalization,
and a combined measure-and-normalize function for audio loudness
conforming to ITU-R BS.1770-4.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

# Type aliases for audio data
MonoAudio = npt.NDArray[np.floating[Any]]  # shape: (samples,)
StereoAudio = npt.NDArray[np.floating[Any]]  # shape: (samples, 2)
SurroundAudio = npt.NDArray[np.floating[Any]]  # shape: (samples, 3-5)
AudioData = npt.NDArray[np.floating[Any]]  # shape: (samples,) or (samples, channels)
FloatArray = npt.NDArray[np.floating[Any]]
Numeric = Union[int, float]


@dataclass
class LoudnessNormalizationResult:
    """Result object for loudness normalization.

    Contains both the normalized audio data and comprehensive
    metadata about the normalization process.

    Parameters
    ----------
    normalized_data : ndarray
        Loudness normalized audio data with same shape and dtype
        as the input data.
    input_loudness : float
        Measured input loudness in dB LUFS.
    target_loudness : float
        Target loudness in dB LUFS that was requested.
    applied_gain : float
        Gain applied to achieve target, in dB.
    applied_gain_linear : float
        Gain applied as a linear multiplier.
    output_loudness : float
        Verified output loudness after normalization in dB LUFS.
    clipped : bool
        True if any samples in the output exceed [-1.0, 1.0] range,
        indicating potential digital clipping.
    """

    normalized_data: FloatArray
    input_loudness: float
    target_loudness: float
    applied_gain: float
    applied_gain_linear: float
    output_loudness: float
    clipped: bool

    def get_stats(self) -> dict:
        """Return a dictionary with loudness normalization statistics.

        Returns
        -------
        dict
            Dictionary containing all loudness normalization metrics
            except the normalized_data array. Suitable for serialization
            to JSON or for logging purposes.

        Examples
        --------
        >>> result = normalize_loudness(data, rate)
        >>> stats = result.get_stats()
        >>> print(f"Input: {stats['input_loudness']:.1f} LUFS")
        >>> print(f"Gain: {stats['applied_gain']:.1f} dB")
        """
        return {
            "input_loudness": self.input_loudness,
            "target_loudness": self.target_loudness,
            "applied_gain": self.applied_gain,
            "applied_gain_linear": self.applied_gain_linear,
            "output_loudness": self.output_loudness,
            "clipped": self.clipped,
        }


def peak(
    data: AudioData,
    target: float,
) -> FloatArray:
    """Peak normalize a signal to a specified peak amplitude.

    Scales the input signal so that the maximum absolute sample value
    matches the specified target level in dB.

    Parameters
    ----------
    data : ndarray
        Input multichannel audio data with floating-point values
        typically in range [-1.0, 1.0]. Shape may be (samples,)
        or (samples, channels).
    target : float
        Desired peak amplitude in dB. For example:
        - 0.0 for full scale
        - -1.0 for -1 dBFS (typical broadcast standard)
        - -3.0 for -3 dBFS

    Returns
    -------
    output : ndarray
        Peak normalized output data with same shape and dtype as input.

    Warns
    -----
    UserWarning
        If any samples in the output reach or exceed full scale (1.0),
        indicating possible clipping.

    Examples
    --------
    >>> import numpy as np
    >>> import pyloudnorm as pyln
    >>> data = np.array([0.1, -0.2, 0.15], dtype=np.float32)
    >>> normalized = pyln.normalize.peak(data, -6.0)
    """
    current_peak: float = float(np.max(np.abs(data)))
    gain: float = np.power(10.0, target / 20.0) / current_peak
    output: FloatArray = gain * data

    if np.max(np.abs(output)) >= 1.0:
        warnings.warn("Possible clipped samples in output.")
    return output


def loudness(
    data: AudioData,
    input_loudness: float,
    target_loudness: float,
) -> FloatArray:
    """Loudness normalize a signal given its measured loudness.

    Applies a gain to scale the audio from its current integrated
    loudness to the target loudness. The input loudness must be
    measured separately (e.g., using Meter.integrated_loudness).

    Parameters
    ----------
    data : ndarray
        Input multichannel audio data with floating-point values.
        Shape may be (samples,) or (samples, channels).
    input_loudness : float
        Pre-measured integrated loudness of the input in dB LUFS.
        Use a Meter instance to obtain this value.
    target_loudness : float
        Target integrated loudness in dB LUFS.
        Common values: -23.0 (EBU R128), -24.0 (ATSC A/85),
        -16.0 (streaming platforms).

    Returns
    -------
    output : ndarray
        Loudness normalized output data with same shape and dtype as input.

    Warns
    -----
    UserWarning
        If any samples in the output reach or exceed full scale (1.0),
        indicating possible clipping.

    Examples
    --------
    >>> import numpy as np
    >>> import pyloudnorm as pyln
    >>> data, rate = ...  # load audio
    >>> meter = pyln.Meter(rate)
    >>> loudness_val = meter.integrated_loudness(data)
    >>> normalized = pyln.normalize.loudness(data, loudness_val, -23.0)
    """
    delta_loudness: float = target_loudness - input_loudness
    gain: float = float(np.power(10.0, delta_loudness / 20.0))
    output: FloatArray = gain * data

    if np.max(np.abs(output)) >= 1.0:
        warnings.warn("Possible clipped samples in output.")
    return output


def normalize_loudness(
    data: AudioData,
    rate: Numeric,
    target_loudness: float = -23.0,
    meter: Optional[Any] = None,
    block_size: float = 0.400,
    filter_class: str = "K-weighting",
    min_duration: float = 0.100,  # Minimum duration in seconds (default 100ms)
) -> LoudnessNormalizationResult:
    """Measure and normalize audio to a target loudness level in one step.

    Combines ITU-R BS.1770-4 loudness measurement with normalization,
    providing comprehensive metadata about the process. This is the
    recommended function for loudness normalization as it handles all
    edge cases and provides detailed feedback.

    Parameters
    ----------
    data : ndarray of float or int
        Input audio data with shape (samples,) for mono or
        (samples, channels) for multi-channel. Channel count must
        not exceed 5. Can be floating point (typically [-1.0, 1.0])
        or integer (int16, int32, etc.). Integer types will be
        automatically converted to float in the range [-1.0, 1.0].
    rate : int or float
        Sampling rate of the audio in Hz. Must be positive.
    target_loudness : float, default=-23.0
        Target integrated loudness in dB LUFS.
        Common values:
        - -23.0 LUFS: EBU R128 (European broadcast)
        - -24.0 LUFS: ATSC A/85 (US broadcast)
        - -16.0 LUFS: Spotify, YouTube
        - -14.0 LUFS: Amazon Music, Tidal
    meter : Meter, optional
        Pre-configured pyloudnorm.Meter instance. If provided,
        its settings override block_size and filter_class.
        If None, a default Meter is created with the specified
        block_size and filter_class.
    block_size : float, default=0.400
        Gating block size in seconds for loudness measurement.
        Standard is 0.400 (400ms) per BS.1770-4. For audio shorter
        than block_size, it will be automatically reduced to fit.
    filter_class : str, default="K-weighting"
        Frequency weighting filter class. Options:
        - "K-weighting" (ITU-R BS.1770-4 standard)
        - "DeMan" (fully compliant alternative)
        - "Fenton/Lee 1" (low complexity variant)
        - "Fenton/Lee 2" (higher complexity variant)
        - "Dash et al." (early modification)
    min_duration : float, default=0.100
        Minimum audio duration in seconds required for processing.
        Audio shorter than this will be returned unchanged with a warning.
        Set to 0 to process any length.

    Returns
    -------
    LoudnessNormalizationResult
        Dataclass with fields:
        - normalized_data: FloatArray
            The loudness-normalized audio data. Returns in the same
            dtype as input (float for float inputs, float32 for integer inputs).
        - input_loudness: float
            Measured input loudness in dB LUFS.
        - target_loudness: float
            The target loudness that was requested.
        - applied_gain: float
            Gain applied in dB (positive = amplification).
        - applied_gain_linear: float
            Gain applied as a linear multiplier.
        - output_loudness: float
            Verified output loudness after normalization in dB LUFS.
        - clipped: bool
            True if output exceeds full scale range.

    Raises
    ------
    ValueError
        If data is not a numpy array, has invalid shape,
        has more than 5 channels, or is empty.
    TypeError
        If rate or target_loudness is not numeric.

    Warns
    -----
    UserWarning
        - If the input is too short for reliable measurement.
        - If the input is silent or too quiet to measure.
        - If the output samples will clip (exceed full scale).

    Notes
    -----
    Integer input data (e.g., int16 from WAV files) is automatically
    converted to float using the appropriate scale factor:
    - int16: divided by 32768.0
    - int32: divided by 2147483648.0
    - int8: divided by 128.0
    - uint8: divided by 255.0 and centered around 0

    For very short audio segments (< block_size), the block_size will be
    automatically reduced to fit the available data. Very short audio
    (< min_duration) will be returned unchanged as loudness measurement
    is unreliable for such short segments.

    The output will be in float32 format for integer inputs to preserve
    precision and avoid the original integer range limitations.

    Examples
    --------
    >>> import numpy as np
    >>> import pyloudnorm as pyln
    >>>
    >>> # Basic usage with float data
    >>> data_float, rate = sf.read("speech.wav")  # returns float
    >>> result = pyln.normalize.normalize_loudness(data_float, rate)
    >>>
    >>> # Works with integer data too
    >>> data_int, rate = sf.read("speech.wav", dtype='int16')
    >>> result = pyln.normalize.normalize_loudness(data_int, rate)
    >>>
    >>> # Normalize to streaming loudness standard
    >>> result = pyln.normalize.normalize_loudness(
    ...     data, rate, target_loudness=-16.0
    ... )
    >>>
    >>> # Short segments are handled gracefully
    >>> short_data = np.random.randn(3200).astype(np.float32)  # 200ms at 16kHz
    >>> result = pyln.normalize.normalize_loudness(short_data, 16000)
    >>>
    >>> # Inspect normalization details
    >>> print(f"Original: {result.input_loudness:.1f} LUFS")
    >>> print(f"Applied gain: {result.applied_gain:.1f} dB")
    >>> if result.clipped:
    ...     print("Warning: Output contains clipping!")
    """
    # --- Input Validation ---
    if not isinstance(data, np.ndarray):
        raise ValueError(f"Data must be numpy.ndarray, got {type(data).__name__}")

    # Save original dtype for output formatting
    input_dtype = data.dtype
    was_integer = np.issubdtype(input_dtype, np.integer)

    # --- Convert integer data to float ---
    if was_integer:
        if input_dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif input_dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif input_dtype == np.int8:
            data = data.astype(np.float32) / 128.0
        elif input_dtype == np.uint8:
            # Center around 0: [0, 255] -> [-1.0, 1.0]
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            # Generic integer conversion using the type's max value
            iinfo = np.iinfo(input_dtype)
            data = data.astype(np.float32) / float(iinfo.max)
    elif not np.issubdtype(input_dtype, np.floating):
        # For other non-float types, try conversion
        try:
            data = data.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Cannot convert {input_dtype} to floating point: {e}")

    # Now validate the float data
    if data.ndim not in [1, 2]:
        raise ValueError(f"Data must be 1D or 2D array, got {data.ndim}D")

    if data.ndim == 2 and data.shape[1] > 5:
        raise ValueError(f"Maximum 5 channels supported, got {data.shape[1]}")

    if not isinstance(rate, (int, float)):
        raise TypeError(f"Rate must be numeric, got {type(rate).__name__}")

    rate_float: float = float(rate)

    if rate_float <= 0:
        raise ValueError(f"Rate must be positive, got {rate}")

    if not isinstance(target_loudness, (int, float)):
        raise TypeError(
            f"target_loudness must be numeric, got {type(target_loudness).__name__}"
        )

    target_loudness_float: float = float(target_loudness)

    # Check for empty data
    if data.size == 0:
        raise ValueError("Data array is empty")

    # --- Handle very short audio ---
    duration = data.shape[0] / rate_float

    if duration < min_duration:
        warnings.warn(
            f"Audio is too short for reliable loudness measurement "
            f"({duration:.3f}s < {min_duration:.3f}s minimum). "
            f"Returning original data unchanged."
        )
        output_data = data.copy()
        if was_integer:
            output_data = output_data.astype(np.float32)

        return LoudnessNormalizationResult(
            normalized_data=output_data,
            input_loudness=float("nan"),  # Cannot measure reliably
            target_loudness=target_loudness_float,
            applied_gain=0.0,
            applied_gain_linear=1.0,
            output_loudness=float("nan"),
            clipped=False,
        )

    # --- Adjust block_size for short audio ---
    effective_block_size = block_size

    # If audio is shorter than block_size, reduce block_size to fit
    if duration < block_size:
        # Use half the duration as block_size to ensure at least 2 blocks
        effective_block_size = min(duration / 2.0, block_size)
        warnings.warn(
            f"Audio duration ({duration:.3f}s) is shorter than requested "
            f"block_size ({block_size:.3f}s). "
            f"Reducing block_size to {effective_block_size:.3f}s for analysis."
        )

    # Ensure block_size is not too small (at least 10ms)
    effective_block_size = max(effective_block_size, 0.010)

    min_samples: int = int(effective_block_size * rate_float)
    if data.shape[0] < min_samples:
        # Even with minimum block_size, audio is too short
        warnings.warn(
            f"Audio is too short for reliable measurement "
            f"({data.shape[0]} samples at {rate}Hz). "
            f"Returning original data unchanged."
        )
        output_data = data.copy()
        if was_integer:
            output_data = output_data.astype(np.float32)

        return LoudnessNormalizationResult(
            normalized_data=output_data,
            input_loudness=float("nan"),
            target_loudness=target_loudness_float,
            applied_gain=0.0,
            applied_gain_linear=1.0,
            output_loudness=float("nan"),
            clipped=False,
        )

    # --- Create Meter if not provided ---
    if meter is None:
        from pyloudnorm.meter import Meter

        meter = Meter(
            rate_float, filter_class=filter_class, block_size=effective_block_size
        )

    # --- Measure Input Loudness ---
    # Make a copy to avoid modifying original data
    work_data: FloatArray = data.copy().astype(np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        input_loudness: float = float(meter.integrated_loudness(work_data))

    # --- Handle Edge Cases ---
    if np.isnan(input_loudness) or np.isinf(input_loudness):
        # Signal is too quiet or silent
        warnings.warn(
            "Input audio is silent or too quiet to measure. "
            "Returning original data unchanged."
        )
        # Return data in original format
        output_data = data.copy()
        if was_integer:
            # Keep float32 format for consistency
            output_data = output_data.astype(np.float32)

        return LoudnessNormalizationResult(
            normalized_data=output_data,
            input_loudness=float("-inf"),
            target_loudness=target_loudness_float,
            applied_gain=0.0,
            applied_gain_linear=1.0,
            output_loudness=float("-inf"),
            clipped=False,
        )

    # --- Compute Gain and Apply ---
    delta_loudness: float = target_loudness_float - input_loudness
    gain_linear: float = float(np.power(10.0, delta_loudness / 20.0))
    gain_db: float = float(20.0 * np.log10(gain_linear))

    # Apply gain to data
    normalized_data: FloatArray = gain_linear * data

    # Keep as float32 for integer inputs (avoids range issues)
    if was_integer:
        normalized_data = normalized_data.astype(np.float32)

    # --- Check for Clipping ---
    peak_value: float = float(np.max(np.abs(normalized_data)))
    clipped: bool = bool(peak_value >= 1.0)

    if clipped:
        peak_db: float = float(20.0 * np.log10(peak_value))
        warnings.warn(
            f"Possible clipping: peak value is {peak_db:.2f} dB. "
            f"Target loudness of {target_loudness_float} LUFS "
            f"may be too high for this audio."
        )

    # --- Verify Output Loudness ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        output_loudness: float = float(
            meter.integrated_loudness(normalized_data.astype(np.float64))
        )

    return LoudnessNormalizationResult(
        normalized_data=normalized_data,
        input_loudness=input_loudness,
        target_loudness=target_loudness_float,
        applied_gain=gain_db,
        applied_gain_linear=gain_linear,
        output_loudness=output_loudness,
        clipped=clipped,
    )
