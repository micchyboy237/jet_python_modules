# jet_python_modules/jet/audio/utils.py   (new file or add to existing utils)
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Literal, Tuple, TypedDict

import numpy as np
import soundfile as sf
from tqdm import tqdm

from jet.audio.record_mic import SAMPLE_RATE  # 16000
from jet.logger import logger

from jet.audio.helpers.silence import calibrate_silence_threshold


def compute_l1_energy(audio_frame: np.ndarray) -> float:
    """
    Mean absolute amplitude of the frame: mean(|x|).

    One-line intuition:
        "Is there sound present, and how strong is it on average?"

    Answers:
        "How active is this signal, ignoring spikes?"

    Notes:
    - Length-independent
    - Robust to noise and peaks
    - Normalized ONLY if input is in [-1.0, 1.0]
    """
    return float(np.mean(np.abs(audio_frame)))


def compute_rms(audio_frame: np.ndarray) -> float:
    """
    Root mean square amplitude of the frame: sqrt(mean(x²)).

    One-line intuition:
        "How loud is this signal on average?"

    Answers:
        "What is the average signal power?"

    Notes:
    - Length-independent
    - Sensitive to peaks
    - Normalized ONLY if input is in [-1.0, 1.0]
    """
    return float(np.sqrt(np.mean(audio_frame * audio_frame)))


def energy(audio_frame: np.ndarray) -> float:
    """
    Total signal energy over the frame: sum(x²).

    One-line intuition:
        "How much sound happened over this entire frame?"

    Answers:
        "How much total power accumulated over time?"

    Notes:
    - Length-dependent
    - Not normalized
    - Frame-size must be fixed to compare values
    """
    return float(np.sum(audio_frame * audio_frame))


def has_sound(
    audio_frame: np.ndarray, 
    threshold: float = 0.01
) -> bool:
    """
    Determine whether an audio frame contains sound above a threshold.

    One-line intuition:
        "Is this frame loud enough to matter?"

    Answers:
        "Should this frame be considered non-silent?"

    Notes:
    - Expects normalized audio in the range [-1.0, 1.0]
    - Typically used with l1_energy or rms
    - Default threshold is tuned for quiet speech detection (~0.01)
    """
    return compute_l1_energy(audio_frame) > threshold


# def has_sound(
#     file_path: str | Path,
#     silence_threshold: float | None = None,
#     chunk_duration: float = 0.25,
#     min_sound_chunks: int = 1,
# ) -> bool:
#     """
#     Determine whether a WAV file contains any detectable speech/sound.
    
#     Parameters
#     ----------
#     file_path : str | Path
#         Path to WAV file
#     silence_threshold : float | None, optional
#         Energy threshold. If None → auto-calibrate using ambient noise.
#     chunk_duration : float, default 0.25
#         Analysis chunk size in seconds (must match live pipeline)
#     min_sound_chunks : int, default 1
#         Minimum number of non-silent chunks required to classify as "has sound"
    
#     Returns
#     -------
#     bool
#         True if at least `min_sound_chunks` chunks exceed the threshold
#     """
#     file_path = Path(file_path)
#     if not file_path.is_file():
#         raise FileNotFoundError(f"Audio file not found: {file_path}")

#     if silence_threshold is None:
#         logger.info("No silence threshold provided → auto-calibrating...")
#         silence_threshold = calibrate_silence_threshold()

#     energies = compute_energies(
#         file_path=file_path,
#         chunk_duration=chunk_duration,
#         silence_threshold=silence_threshold,
#     )

#     sound_chunks = [e for e in energies if not e.get("is_silent", False)]
#     has_detected_sound = len(sound_chunks) >= min_sound_chunks

#     logger.info(
#         f"has_sound('{file_path.name}'): {len(sound_chunks)} sound chunk(s) "
#         f"(≥ {silence_threshold:.6f}), threshold used: {silence_threshold:.6f} → {has_detected_sound}"
#     )
#     return has_detected_sound


def compute_energy(audio_frame: np.ndarray) -> float:
    """Return mean absolute amplitude of an audio frame."""
    return float(np.mean(np.abs(audio_frame)))


def compute_energies(
    file_path: str | Path,
    chunk_duration: float = 0.5,
    silence_threshold: float | None = None,
) -> List[Dict[str, float]]:
    """
    Load a WAV file and compute the same per-chunk energy that the live streaming pipeline uses.

    Parameters
    ----------
    file_path : str | Path
        Path to a mono or stereo WAV file (any sample rate – will be resampled to 16 kHz).
    chunk_duration : float, default 0.5
        Size of each analysis block in seconds (matches the live stream default).
    silence_threshold : float | None
        If provided, also returns whether each chunk is considered silent.

    Returns
    -------
    List[Dict[str, float]]
        Each dict contains:
        - start_s : start time of the chunk (seconds)
        - end_s   : end time of the chunk (seconds)
        - energy  : mean absolute amplitude of the chunk
        - is_silent (optional) : True if energy < silence_threshold
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)               # downmix to mono
    if sr != SAMPLE_RATE:
        logger.debug(f"Resampling {file_path.name} from {sr} → {SAMPLE_RATE} Hz")
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    audio = audio.astype(np.float32)
    if np.abs(audio).max() > 1.0:                # safety normalize
        audio /= np.abs(audio).max()

    chunk_samples = int(SAMPLE_RATE * chunk_duration)
    total_chunks = int(np.ceil(len(audio) / chunk_samples))

    results: List[Dict[str, float]] = []

    with tqdm(total=total_chunks, desc="Energy chunks", unit="chunk", leave=False) as pbar:
        for i in range(total_chunks):
            start_sample = i * chunk_samples
            end_sample = min((i + 1) * chunk_samples, len(audio))
            chunk = audio[start_sample:end_sample]

            energy = compute_energy(chunk)

            item: Dict[str, float] = {
                "start_s": round(start_sample / SAMPLE_RATE, 3),
                "end_s":   round(end_sample   / SAMPLE_RATE, 3),
                "energy":  round(energy, 6),
            }
            if silence_threshold is not None:
                item["is_silent"] = energy < silence_threshold

            results.append(item)
            pbar.update(1)

    logger.info(f"Computed energy for {len(results)} chunks from {file_path.name}")
    return results


def detect_sound(audio_chunk: np.ndarray, threshold: float) -> bool:
    """
    Detect if an audio chunk contains audible sound (non-silence).
    
    Parameters
    ----------
    audio_chunk : np.ndarray
        Raw audio samples (float32, normalized to [-1.0, 1.0])
    threshold : float
        Silence energy threshold (higher = stricter)
    
    Returns
    -------
    bool
        True if chunk has detectable sound (energy >= threshold)
    """
    energy = compute_energy(audio_chunk)
    is_sound = energy >= threshold
    logger.debug(
        f"detect_sound → energy: {energy:.6f}, threshold: {threshold:.6f}, has_sound: {is_sound}"
    )
    return is_sound


# Single source of truth for loudness label literals
LoudnessLabel = Literal[
    "Very Quiet",
    "Quiet",
    "Soft",
    "Normal",
    "Loud",
    "Raised",
    "Very Loud",
    "Extremely Loud",
    "Unknown",
]


def rms_to_loudness_label(rms_norm: float) -> LoudnessLabel:
    if rms_norm < 0.004:
        return "Very Quiet"
    if rms_norm < 0.015:
        return "Quiet"
    if rms_norm < 0.060:
        return "Soft"
    if rms_norm < 0.140:
        return "Normal"
    if rms_norm < 0.260:
        return "Loud"
    if rms_norm < 0.380:
        return "Raised"           # ← new intermediate label (optional rename)
    if rms_norm < 0.480:
        return "Very Loud"
    return "Extremely Loud"


def rms_to_loudness_labels(
    energies: List[float],
) -> Tuple[List[LoudnessLabel], dict]:
    """
    Convert RMS energies to human-readable loudness labels
    using percentile-based normalization.
    """
    arr = np.asarray(energies, dtype=float)
    percentiles = np.percentile(arr, [5, 20, 40, 60, 80, 95])

    bins = [
        (-np.inf, percentiles[0], "Very Quiet"),
        (percentiles[0], percentiles[1], "Quiet"),
        (percentiles[1], percentiles[2], "Soft"),
        (percentiles[2], percentiles[3], "Normal"),
        (percentiles[3], percentiles[4], "Loud"),
        (percentiles[4], percentiles[5], "Very Loud"),
        (percentiles[5], np.inf, "Extremely Loud"),
    ]

    labels: List[LoudnessLabel] = []
    for v in arr:
        for lo, hi, label in bins:
            if lo <= v < hi:
                labels.append(label)  # type: ignore
                break

    metadata = {
        "percentile_thresholds": {
            "p5": float(percentiles[0]),
            "p20": float(percentiles[1]),
            "p40": float(percentiles[2]),
            "p60": float(percentiles[3]),
            "p80": float(percentiles[4]),
            "p95": float(percentiles[5]),
        }
    }

    return labels, metadata


@dataclass(frozen=True)
class SegmentLike:           # ← no Protocol needed here
    start_frame: int
    end_frame: int


class SegmentLoudnessResult(TypedDict):
    segment_index: int
    loudness: LoudnessLabel


def segment_loudness_median_label(
    segments: List[SegmentLike],
    frame_labels: List[LoudnessLabel],
) -> List[SegmentLoudnessResult]:
    """
    Assign a loudness label to each segment using the most frequent
    (median) frame-level label.
    """
    results: List[SegmentLoudnessResult] = []
    max_len = len(frame_labels)

    for idx, seg in enumerate(segments):
        start = max(0, seg.start_frame)
        end = min(seg.end_frame, max_len)
        labels = frame_labels[start:end]

        if not labels:
            loudness: LoudnessLabel = "Unknown"
        else:
            counts = Counter(labels)
            loudness = counts.most_common(1)[0][0]

        results.append({
            "segment_index": idx,
            "loudness": loudness,
        })

    return results


def segment_loudness_energy_weighted(
    segments: List[SegmentLike],
    frame_labels: List[LoudnessLabel],
    frame_energies: List[float],
) -> List[SegmentLoudnessResult]:
    """
    Assign a loudness label to each segment weighted by RMS energy.

    The label whose frames contribute the most total energy
    is selected as the segment loudness.
    """
    results: List[SegmentLoudnessResult] = []
    max_len = min(len(frame_labels), len(frame_energies))

    for idx, seg in enumerate(segments):
        start = max(0, seg.start_frame)
        end = min(seg.end_frame, max_len)
        energy_by_label: Dict[LoudnessLabel, float] = defaultdict(float)

        for lbl, eng in zip(frame_labels[start:end], frame_energies[start:end]):
            energy_by_label[lbl] += eng

        if not energy_by_label:
            loudness: LoudnessLabel = "Unknown"
        else:
            loudness = max(
                energy_by_label.items(),
                key=lambda item: item[1],
            )[0]

        results.append({
            "segment_index": idx,
            "loudness": loudness,
        })

    return results


if __name__ == "__main__":
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/silero/generated/silero_vad_stream/segment_001/sound.wav"

    threshold = calibrate_silence_threshold()
    energies = compute_energies(audio_file, silence_threshold=threshold)

    for e in energies[:10]:
        print(e)
    # → {'start_s': 0.0, 'end_s': 0.5, 'energy': 0.012345, 'is_silent': True}
