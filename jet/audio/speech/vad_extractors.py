import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from jet.audio.audio_types import AudioInput
from jet.audio.audio_waveform.vad._types import SpeechSegment

# from jet.audio.audio_waveform.vad.vad_speech_segments_extractor import (
#     extract_speech_timestamps,
# )
from jet.audio.audio_waveform.vad.vad_firered import extract_speech_timestamps
from jet.audio.audio_waveform.vad.vad_firered_hybrid import FireRedVAD
from jet.audio.helpers.config import FRAME_SHIFT_MS, SAMPLE_RATE, SILENCE_MAX_THRESHOLD
from jet.audio.helpers.energy_base import trim_silent_frames
from jet.audio.speech._main_vad_extractors import main
from jet.audio.speech.vad_loaders import load_vad_hybrid_probs
from jet.audio.speech.vad_types import (
    TroughToTroughSegment,
    VADSegment,
    ValleyInfo,
    ValleyTrough,
)
from jet.audio.utils.loader import load_audio
from rich.console import Console

console = Console()

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
}


def is_probs_list(obj):
    """Returns True if obj is a list of floats."""
    return (
        isinstance(obj, list)
        and len(obj) > 0
        and all(isinstance(x, float) for x in obj)
    )


def _linkify(path: Path):
    return f"[link=file://{path}]{path.name}[/link]"


def load_probs(
    probs_or_audio: list[float] | AudioInput, default_audio: str | Path | None = None
) -> tuple[list[float], Optional[np.ndarray]]:
    """
    Attempts to load probability scores for VAD either from a list of floats,
    from an AudioInput (str path, bytes, os.PathLike, ndarray, or Tensor),
    or as a JSON string. Falls back on default_audio if parsing fails.

    Returns:
        probs (list[float]): VAD probability scores.
        audio_np (Optional[np.ndarray]): Decoded waveform if AudioInput required
            audio loading, otherwise None.

    Raises:
        ValueError: If unable to parse input and no default_audio is provided.
    """
    input_value = probs_or_audio
    audio_np: Optional[np.ndarray] = None

    # Case 1: Already a list of floats.
    if is_probs_list(input_value):
        return input_value, None

    # Case 2: Path (str | Path): .npy/.json/.txt = scores, otherwise treat as audio.
    if isinstance(input_value, (str, Path)):
        input_path = Path(input_value)
        if input_path.is_file():
            ext = input_path.suffix.lower()
            if ext == ".npy":
                console.print(f"Loading probabilities from: {_linkify(input_path)}")
                np_load = np.load(input_path, allow_pickle=True)
                probs = np_load.tolist() if isinstance(np_load, np.ndarray) else np_load
                if not is_probs_list(probs):
                    raise ValueError(
                        f".npy file does not contain a list of floats: {_linkify(input_path)}"
                    )
                return probs, None
            elif ext in {".json", ".txt"}:
                console.print(f"Loading probabilities from: {_linkify(input_path)}")
                with open(input_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if is_probs_list(loaded):
                    return loaded, None
                else:
                    raise ValueError(
                        f"JSON file is not a list of floats: {_linkify(input_path)}"
                    )
            else:
                console.print(f"Loading audio from: {_linkify(input_path)}")
                audio = load_audio(input_path)
                audio_np = (
                    audio[0] if isinstance(audio, tuple) and len(audio) == 2 else audio
                )
                _, probs = extract_speech_timestamps(
                    audio=audio_np,
                    threshold=0.5,
                    min_speech_duration_sec=0.250,
                    min_silence_duration_sec=0.250,
                    with_scores=True,
                    use_hybrid=True,
                )
                if not is_probs_list(probs):
                    raise ValueError(
                        f"Extracted VAD scores are not a list of floats from: {_linkify(input_path)}"
                    )
                return probs, audio_np

    # Case 3: JSON string containing scores
    if isinstance(input_value, str):
        try:
            loaded = json.loads(input_value)
            if is_probs_list(loaded):
                return loaded, None
        except Exception:
            pass

    # Case 4: AudioInput objects - bytes, ndarray, Tensor
    # Handle numpy.ndarray: treat as waveform
    if isinstance(input_value, np.ndarray):
        audio_np = input_value
        _, probs = extract_speech_timestamps(
            audio=audio_np,
            threshold=0.5,
            min_speech_duration_sec=0.250,
            min_silence_duration_sec=0.250,
            with_scores=True,
        )
        if not is_probs_list(probs):
            raise ValueError(
                "Extracted VAD scores are not a list of floats from ndarray input."
            )
        return probs, audio_np

    # Handle bytes: treat as audio file bytes
    if isinstance(input_value, bytes):
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(input_value)
            tmp.flush()
            audio = load_audio(tmp.name)
            audio_np = (
                audio[0] if isinstance(audio, tuple) and len(audio) == 2 else audio
            )
            _, probs = extract_speech_timestamps(
                audio=audio_np,
                threshold=0.5,
                min_speech_duration_sec=0.250,
                min_silence_duration_sec=0.250,
                with_scores=True,
            )
            if not is_probs_list(probs):
                raise ValueError(
                    "Extracted VAD scores are not a list of floats from bytes input."
                )
            return probs, audio_np

    # Handle Torch Tensor (optional, duck-type)
    try:
        import torch

        if isinstance(input_value, torch.Tensor):
            audio_np = input_value.detach().cpu().numpy()
            _, probs = extract_speech_timestamps(
                audio=audio_np,
                threshold=0.5,
                min_speech_duration_sec=0.250,
                min_silence_duration_sec=0.250,
                with_scores=True,
            )
            if not is_probs_list(probs):
                raise ValueError(
                    "Extracted VAD scores are not a list of floats from Tensor input."
                )
            return probs, audio_np
    except ImportError:
        pass

    # If we haven't parsed, fall back to default_audio if provided
    if default_audio is not None:
        console.print(
            f"[yellow]Input not recognized, falling back to default audio: "
            f"{_linkify(Path(default_audio))}[/yellow]"
        )
        audio = load_audio(default_audio)
        audio_np = audio[0] if isinstance(audio, tuple) and len(audio) == 2 else audio
        _, probs = extract_speech_timestamps(
            audio=audio_np,
            threshold=0.5,
            min_speech_duration_sec=0.250,
            min_silence_duration_sec=0.250,
            with_scores=True,
        )
        if not is_probs_list(probs):
            raise ValueError(
                "Extracted VAD scores are not a list of floats from fallback audio."
            )
        return probs, audio_np

    raise ValueError("Input could not be parsed and no default_audio was provided.")


def base_extract_valley_troughs(
    valleys: List[VADSegment], duration_s: float = 0.25
) -> List[ValleyTrough]:
    """
    Extracts the lowest-probability frames (troughs) from a list of VADSegment
    valleys, but only includes valleys that have exactly one trough and
    duration >= duration_s.
    """
    filtered_valleys = [
        valley
        for valley in valleys
        if len(valley["details"].get("troughs", [])) == 1
        and valley["duration_s"] >= duration_s
    ]
    last_frame = max((v["frame_end"] for v in filtered_valleys), default=-1)
    valley_troughs: List[ValleyTrough] = []
    for valley in filtered_valleys:
        details = valley["details"]
        valley_score = details.get("valley_score", 0.0)
        trough_score = details.get("trough_score", 0.0)
        final_score = details.get("final_score", 0.0)
        valley_info: ValleyInfo = {
            "frame_start": valley["frame_start"],
            "frame_end": valley["frame_end"],
            "frame_length": valley["frame_length"],
            "start_s": valley["start_s"],
            "end_s": valley["end_s"],
            "duration_s": valley["duration_s"],
            "valley_score": valley_score,
            "trough_score": trough_score,
            "final_score": final_score,
            "global_frame_start": valley["frame_start"],
            "global_frame_end": valley["frame_end"],
            "global_start_s": valley["start_s"],
            "global_end_s": valley["end_s"],
            "global_duration_s": valley["duration_s"],
            "global_valley_score": valley_score,
            "global_trough_score": trough_score,
            "global_final_score": final_score,
            "is_last": valley["frame_end"] >= last_frame,
        }
        valley_troughs.append(
            ValleyTrough(
                frame=details["min_prob_frame"],
                global_frame=details["min_prob_frame"],
                prob=details["min_probability"],
                time_s=details["min_prob_s"],
                global_time_s=details["min_prob_s"],
                valley=valley_info,
            )
        )
    return valley_troughs


def get_best_valley_trough(
    probs_or_audio: List[float] | AudioInput,
    sample_rate: int = SAMPLE_RATE,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    smoothing_window: int = 20,
    trough_height: Optional[float] = None,
    trough_prominence: float = 0.15,
    trough_distance: int = 5,
    valley_threshold: Optional[float] = None,
    min_valley_duration_s: float = 0.8,
    min_valley_frames: Optional[int] = None,
    frame_offset: int = 0,
    min_trough_offset_s: float = 0.4,
) -> Optional[ValleyTrough]:
    """
    Returns the single best ValleyTrough based on the highest final_score.
    Returns None if no suitable trough is found.

    Args:
        probs_or_audio: VAD probabilities as a list[float], or an AudioInput
            (str, bytes, os.PathLike, ndarray, or Tensor) that load_probs
            can resolve into probabilities.
        All other kwargs are forwarded to extract_valley_troughs.
    """
    probs, _ = load_probs(probs_or_audio)
    all_troughs = extract_valley_troughs(
        probs_or_audio=probs,
        sample_rate=sample_rate,
        frame_shift_ms=frame_shift_ms,
        smoothing_window=smoothing_window,
        trough_height=trough_height,
        trough_prominence=trough_prominence,
        trough_distance=trough_distance,
        valley_threshold=valley_threshold,
        min_valley_duration_s=min_valley_duration_s,
        min_valley_frames=min_valley_frames,
        frame_offset=frame_offset,
        min_trough_offset_s=min_trough_offset_s,
    )
    if not all_troughs:
        return None
    best = max(all_troughs, key=lambda t: t["valley"].get("final_score", 0.0))
    return best


def get_last_valley_trough(
    probs_or_audio: List[float] | AudioInput,
    sample_rate: int = SAMPLE_RATE,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    smoothing_window: int = 20,
    trough_height: Optional[float] = None,
    trough_prominence: float = 0.15,
    trough_distance: int = 5,
    valley_threshold: Optional[float] = None,
    min_valley_duration_s: float = 0.8,
    min_valley_frames: Optional[int] = None,
    frame_offset: int = 0,
    min_trough_offset_s: float = 0.4,
) -> Optional[ValleyTrough]:
    """
    Return the valley trough whose valley covers the last audio frame
    (i.e. ``valley.is_last == True``).

    "Covers the last frame" means the valley's ``frame_end`` reaches
    ``len(probs) - 1`` (accounting for the ``frame_offset`` shift).
    If more than one such trough exists, the one with the highest
    ``final_score`` is returned.  Returns ``None`` when no matching
    trough is found.

    Args:
        probs_or_audio: VAD probabilities as a list[float], or an AudioInput
            (str, bytes, os.PathLike, ndarray, or Tensor) that load_probs
            can resolve into probabilities.
        All other kwargs are forwarded to extract_valley_troughs.
    """
    probs, _ = load_probs(probs_or_audio)
    all_troughs = extract_valley_troughs(
        probs_or_audio=probs,
        sample_rate=sample_rate,
        frame_shift_ms=frame_shift_ms,
        smoothing_window=smoothing_window,
        trough_height=trough_height,
        trough_prominence=trough_prominence,
        trough_distance=trough_distance,
        valley_threshold=valley_threshold,
        min_valley_duration_s=min_valley_duration_s,
        min_valley_frames=min_valley_frames,
        frame_offset=frame_offset,
        min_trough_offset_s=min_trough_offset_s,
    )
    last_troughs = [t for t in all_troughs if t["valley"]["is_last"]]
    if not last_troughs:
        return None
    return max(last_troughs, key=lambda t: t["valley"].get("final_score", 0.0))


# Each half is either (probs, trough) when the input was already a list[float],
# or (probs, trough, audio_np) when the input was an AudioInput (str, bytes,
# os.PathLike, ndarray, or Tensor) that had to be decoded into probabilities —
# so callers can access the raw waveform without re-loading.
SplitResult = Union[
    Tuple[List[float], List[float], ValleyTrough],
    Tuple[List[float], List[float], ValleyTrough, np.ndarray, np.ndarray],
]


def split_best_valley_trough(
    probs_or_audio: List[float] | AudioInput,
    sample_rate: int = SAMPLE_RATE,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    smoothing_window: int = 20,
    trough_height: Optional[float] = 0.3,
    trough_prominence: float = 0.0,
    trough_distance: int = 5,
    valley_threshold: Optional[float] = None,
    min_valley_duration_s: float = 0.1,
    min_valley_frames: Optional[int] = None,
    frame_offset: int = 0,
    min_trough_offset_s: float = 0.4,
    trim_silence: bool = True,
    trim_threshold: Optional[float] = SILENCE_MAX_THRESHOLD,
) -> Optional[SplitResult]:
    """
    Split a VAD probability list or audio into two halves at the best valley trough,
    then trim trailing silence from the left half and leading silence from the right half.

    Parameters
    ----------
    trim_silence : bool
        If True, strip silent frames from the inner edges of each half after splitting.
    trim_threshold : float, optional
        VAD probability below which a frame is considered silent for trimming purposes.
        Defaults to ``valley_threshold`` if set, otherwise ``trough_height``, otherwise 0.3.
    """
    probs, data = load_vad_hybrid_probs(probs_or_audio)
    audio_np = data["audio_np"]

    best_trough = get_best_valley_trough(
        probs_or_audio=probs,
        sample_rate=sample_rate,
        frame_shift_ms=frame_shift_ms,
        smoothing_window=smoothing_window,
        trough_height=trough_height,
        trough_prominence=trough_prominence,
        trough_distance=trough_distance,
        valley_threshold=valley_threshold,
        min_valley_duration_s=min_valley_duration_s,
        min_valley_frames=min_valley_frames,
        frame_offset=frame_offset,
        min_trough_offset_s=min_trough_offset_s,
    )
    if best_trough is None:
        return None

    split_frame: int = best_trough["frame"]
    left_probs: List[float] = probs[:split_frame]
    right_probs: List[float] = probs[split_frame:]

    if audio_np is not None:
        seconds_per_frame = frame_shift_ms / 1000.0
        split_sample = int(split_frame * seconds_per_frame * sample_rate)
        split_sample = max(0, min(split_sample, len(audio_np)))
        left_audio: Optional[np.ndarray] = audio_np[:split_sample]
        right_audio: Optional[np.ndarray] = audio_np[split_sample:]
    else:
        left_audio = None
        right_audio = None

    if trim_silence:
        left_probs, left_audio = trim_silent_frames(
            left_probs,
            left_audio,
            trim_left=False,
            trim_right=True,
            frame_shift_ms=frame_shift_ms,
            sample_rate=sample_rate,
        )
        right_probs, right_audio = trim_silent_frames(
            right_probs,
            right_audio,
            trim_left=True,
            trim_right=False,
            frame_shift_ms=frame_shift_ms,
            sample_rate=sample_rate,
        )

    if audio_np is not None:
        return left_probs, right_probs, best_trough, left_audio, right_audio
    return left_probs, right_probs, best_trough


def extract_valley_troughs(
    probs_or_audio: List[float] | AudioInput,
    sample_rate: int = SAMPLE_RATE,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    smoothing_window: int = 0,
    trough_height: Optional[float] = None,
    trough_prominence: float = 0.15,
    trough_distance: int = 5,
    valley_threshold: Optional[float] = None,
    min_valley_duration_s: float = 0.25,
    min_valley_frames: Optional[int] = None,
    frame_offset: int = 0,
    min_trough_offset_s: float = 0.4,
) -> List[ValleyTrough]:
    """
    Extract salient valley troughs (composite valley + trough detection) with scoring.

    Args:
        probs_or_audio: VAD speech probabilities as a list[float] in [0.0, 1.0],
            or an AudioInput (str, bytes, os.PathLike, ndarray, or Tensor)
            that load_probs can resolve into probabilities.
        sample_rate: Audio sample rate in Hz.
        frame_shift_ms: Frame shift in milliseconds.
        smoothing_window: Smoothing window size (0 = disabled).
        trough_height: Min trough height (None = auto).
        trough_prominence: Trough prominence. Default 0.15.
        trough_distance: Min frames between troughs. Default 5.
        valley_threshold: Valley threshold (None = auto).
        min_valley_duration_s: Minimum valley duration. Default 0.25.
        min_valley_frames: Min valley frames (overrides duration if set).
        frame_offset: Global frame offset for chunked processing.
        min_trough_offset_s: Min seconds from start for valid trough. Default 0.4.

    Returns:
        List[ValleyTrough]: Detected troughs with enclosing valley info,
        local/global coordinates, and composite scores (valley_score × trough_score).
    """
    from jet.audio.speech.vad_peak_analyzer import VADPeakAnalyzer

    probs, _ = load_probs(probs_or_audio)
    analyzer = VADPeakAnalyzer(
        sample_rate=sample_rate,
        frame_shift_ms=frame_shift_ms,
    )
    smoothed = (
        smooth_vad_probs(probs, window=smoothing_window) if smoothing_window else probs
    )
    troughs = analyzer.extract_troughs(
        smoothed,
        height=trough_height,
        prominence=trough_prominence,
        distance=trough_distance,
    )
    valleys = analyzer.extract_valleys(
        smoothed,
        threshold=valley_threshold,
        troughs=troughs,
    )
    valleys = analyzer.filter_short_segments(
        valleys,
        min_duration_s=min_valley_duration_s,
        min_duration_frames=min_valley_frames,
    )
    for valley in valleys:
        details = valley["details"]
        valley_score = compute_valley_score(
            min_prob=details.get("min_probability", 1.0),
            mean_prob=details.get("mean_probability", 1.0),
            duration_s=valley["duration_s"],
        )
        details["valley_score"] = valley_score
        trough_list = details.get("troughs", [])
        if trough_list:
            trough = min(
                trough_list,
                key=lambda t: t.get("details", {}).get("trough_probability", 1.0),
            )
            t_details = trough.get("details", {})
            trough_score = compute_trough_score(
                min_prob=t_details.get("trough_probability", 1.0),
                prominence=t_details.get("prominence", 0.0),
                width=t_details.get("width", 0.0),
            )
            final_score = valley_score * trough_score
            details["trough_score"] = trough_score
            details["final_score"] = final_score
        else:
            details["trough_score"] = 0.0
            details["final_score"] = 0.0

    filtered_valleys = [
        v
        for v in valleys
        if len(v.get("details", {}).get("troughs", [])) == 1
        and v["duration_s"] >= min_valley_duration_s
    ]

    result: List[ValleyTrough] = []
    seconds_per_frame = frame_shift_ms / 1000.0
    total_frames = len(probs)

    for valley in filtered_valleys:
        details = valley["details"]
        local_trough_time_s = details["min_prob_s"]
        if local_trough_time_s < min_trough_offset_s:
            continue
        global_trough_time_s = local_trough_time_s + (frame_offset * seconds_per_frame)
        valley_info: ValleyInfo = {
            "frame_start": valley["frame_start"],
            "frame_end": valley["frame_end"],
            "frame_length": valley["frame_length"],
            "start_s": valley["start_s"],
            "end_s": valley["end_s"],
            "duration_s": valley["duration_s"],
            "valley_score": details["valley_score"],
            "trough_score": details["trough_score"],
            "final_score": details["final_score"],
            "global_frame_start": valley["frame_start"] + frame_offset,
            "global_frame_end": valley["frame_end"] + frame_offset,
            "global_start_s": valley["start_s"] + (frame_offset * seconds_per_frame),
            "global_end_s": valley["end_s"] + (frame_offset * seconds_per_frame),
            "global_duration_s": valley["duration_s"],
            "global_valley_score": details["valley_score"],
            "global_trough_score": details["trough_score"],
            "global_final_score": details["final_score"],
            "is_last": valley["frame_end"] >= total_frames - 1,
        }
        result.append(
            {
                "frame": details["min_prob_frame"],
                "global_frame": details["min_prob_frame"] + frame_offset,
                "prob": details["min_probability"],
                "time_s": local_trough_time_s,
                "global_time_s": global_trough_time_s,
                "valley": valley_info,
            }
        )
    return result


def extract_valley_troughs_from_np_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    vad_threshold: float = 0.3,
    min_speech_duration_sec: float = 0.25,
    min_silence_duration_sec: float = 0.25,
    smoothing_window: int = 20,
    frame_offset: int = 0,
    min_trough_offset_s: float = 0.4,
    min_valley_duration_s: float = 0.25,
    temp_dir: str | Path | None = None,
    trough_height: float | None = None,
    trough_prominence: float = 0.15,
    trough_distance: int = 5,
    vad: FireRedVAD | None = None,
) -> list[ValleyTrough]:
    """
    Extract valley troughs (strong silence positions) from a raw numpy audio clip.

    This is a high-level utility that computes speech probability curves using
    a VAD, then analyzes the result to return a list of the most prominent
    troughs located in sufficiently silent zones. Suitable for downstream
    alignment, trimming, splitting, etc.

    Workflow:
        1. Saves the provided audio (float32, 16kHz recommended) to a temp WAV.
        2. Runs extract_speech_timestamps (FireRed VAD) to obtain framewise
           speech probabilities.
        3. Runs extract_valley_troughs on those probabilities.
        4. Returns the troughs list, each with local and global info.
        5. Always removes the temporary WAV file (even on error/exit).

    Args:
        audio: 1D numpy array of the audio waveform (float32/float64 preferred).
        sample_rate: Sampling rate of audio (Hz).
        vad_threshold: Threshold for considering a frame as speech.
        min_speech_duration_sec: Minimum seconds required for a speech segment.
        min_silence_duration_sec: Minimum required silence (sec) between segments.
        smoothing_window: Smoothing window (frames) for VAD probability smoothing.
        frame_offset: Frame index offset for adjusting global/local outputs.
        min_trough_offset_s: Min time since start before a trough is eligible.
        temp_dir: Optional path for temporary WAV file; defaults to system temp.
        trough_height: Optional minimum height threshold for trough detection.
        trough_prominence: Minimum prominence for detected troughs.
        trough_distance: Minimum distance (frames) between detected troughs.

    Returns:
        List of ValleyTrough dicts, or an empty list on failure.
    """
    if len(audio) == 0:
        return []
    audio = np.asarray(audio, dtype=np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

    with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=False) as tmp:
        temp_wav_path = Path(tmp.name)
    try:
        sf.write(str(temp_wav_path), audio, sample_rate, subtype="FLOAT")
        _, probs = extract_speech_timestamps(
            audio=str(temp_wav_path),
            threshold=vad_threshold,
            min_speech_duration_sec=min_speech_duration_sec,
            min_silence_duration_sec=min_silence_duration_sec,
            with_scores=True,
            vad=vad,
        )
        if not probs:
            return []
        troughs = extract_valley_troughs(
            probs_or_audio=probs,
            smoothing_window=smoothing_window,
            frame_offset=frame_offset,
            min_trough_offset_s=min_trough_offset_s,
            min_valley_duration_s=min_valley_duration_s,
            frame_shift_ms=frame_shift_ms,
            trough_height=trough_height,
            trough_prominence=trough_prominence,
            trough_distance=trough_distance,
        )
        return troughs
    finally:
        try:
            if temp_wav_path.exists():
                temp_wav_path.unlink()
        except Exception:
            pass


def extract_trough_to_trough(
    probs_or_audio: List[float] | AudioInput,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    sample_rate: int = SAMPLE_RATE,
    with_audio: bool = False,
    with_scores: bool = False,
) -> Union[
    List[TroughToTroughSegment],
    List[Tuple[TroughToTroughSegment, np.ndarray]],
    Tuple[List[TroughToTroughSegment], List[float]],
    Tuple[List[Tuple[TroughToTroughSegment, np.ndarray]], List[float]],
]:
    """
    Create segments spanning from one valley trough to the next.

    This function automatically:
    1. Loads/resolves VAD probabilities from the input (audio file, numpy array, list of floats, etc.)
    2. Extracts valley troughs using default parameters
    3. Creates segments between consecutive troughs (including start-to-first and last-to-end)

    For N valley_troughs, this produces N+1 segments:
        segment_0: t=0          → trough[0]
        segment_1: trough[0]    → trough[1]
        ...
        segment_N: trough[N-1]  → end of audio

    When with_scores=True, each TroughToTroughSegment is also populated with:
    - segment_probs: list of VAD probabilities for this segment
    - prob_stats: statistics (mean, max, min, std, median) of those probabilities
    - segments: list of SpeechSegment objects from extract_speech_timestamps that
      fall within this trough-to-trough segment's time range

    Args:
        probs_or_audio: VAD probabilities as a list[float], or an AudioInput
            (str, bytes, os.PathLike, ndarray, or Tensor) that load_probs
            can resolve into probabilities.
        frame_shift_ms: Frame shift in milliseconds.
        sample_rate: Audio sample rate in Hz (used for sample-to-time conversion).
        with_audio: If True, return list of (segment, audio_slice) tuples.
                   Audio is extracted from the input if it's an audio source.
        with_scores: If True, include per-segment VAD probability scores and
                    statistics (mean, max, min, std, median) in each segment,
                    as well as the list of SpeechSegment objects that overlap
                    with each trough-to-trough segment.
                    When with_audio is also True, returns (segments_with_audio, probs).
                    When with_audio is False, returns (segments, probs).

    Returns:
        If with_audio=False, with_scores=False: List[TroughToTroughSegment]
        If with_audio=True, with_scores=False: List[Tuple[TroughToTroughSegment, np.ndarray]]
        If with_audio=False, with_scores=True: Tuple[List[TroughToTroughSegment], List[float]]
        If with_audio=True, with_scores=True: Tuple[List[Tuple[TroughToTroughSegment, np.ndarray]], List[float]]
        Each audio slice is a numpy array of the waveform for that segment.

    Logs:
        Logs the number of troughs processed, segments created, and audio slicing info.
        When with_scores is True, logs probability statistics for each segment
        and the number of SpeechSegment objects assigned to each trough-to-trough segment.
    """
    probs, audio_np = load_probs(probs_or_audio)
    if not probs:
        console.print(
            "[yellow]extract_trough_to_trough: no probabilities extracted, returning empty list.[/yellow]"
        )
        if with_scores:
            return ([], probs) if not with_audio else (([], []), probs)
        return [] if not with_audio else []

    valley_troughs = extract_valley_troughs(
        probs_or_audio=probs,
        sample_rate=sample_rate,
        frame_shift_ms=frame_shift_ms,
    )
    if not valley_troughs:
        console.print(
            "[yellow]extract_trough_to_trough: no valley_troughs found, returning empty list.[/yellow]"
        )
        if with_scores:
            return ([], probs) if not with_audio else (([], []), probs)
        return [] if not with_audio else []

    if with_audio and audio_np is None:
        raise ValueError(
            "extract_trough_to_trough: with_audio=True requires an audio input "
            "(not just probabilities). Provide an audio file path, numpy array, etc."
        )

    # When with_scores=True, also get the speech segments from extract_speech_timestamps
    # so we can populate the 'segments' field on each TroughToTroughSegment
    all_speech_segments: Optional[List[SpeechSegment]] = None
    if with_scores:
        console.print(
            "[cyan]extract_trough_to_trough: with_scores=True, extracting speech segments[/cyan]"
        )
        # We need to call extract_speech_timestamps on the original audio to get segments
        # If we already have audio_np from load_probs, use it; otherwise try the original input
        source_audio = probs_or_audio
        if audio_np is not None:
            source_audio = audio_np

        try:
            # IMPORTANT: Use return_seconds=True so segment start/end are in seconds
            # This allows proper comparison with our start_s/end_s which are also in seconds
            result = extract_speech_timestamps(
                audio=source_audio,
                threshold=0.5,
                min_speech_duration_sec=0.250,
                min_silence_duration_sec=0.250,
                return_seconds=True,  # <-- KEY FIX: return values in seconds
                with_scores=True,
            )
            # When with_scores=True, extract_speech_timestamps returns Tuple[List[SpeechSegment], List[float]]
            if isinstance(result, tuple) and len(result) == 2:
                all_speech_segments, _ = result
                console.print(
                    f"[green]extract_trough_to_trough: got {len(all_speech_segments)} speech segments "
                    f"from extract_speech_timestamps[/green]"
                )
                # Log first few segments for debugging
                for i, seg in enumerate(all_speech_segments[:3]):
                    console.print(
                        f"  [dim]Speech seg {i}: start={seg['start']:.3f}s, end={seg['end']:.3f}s, "
                        f"type={seg['type']}, duration={seg['duration']:.3f}s[/dim]"
                    )
        except Exception as e:
            console.print(
                f"[yellow]extract_trough_to_trough: failed to extract speech segments: {e}. "
                f"segments field will be empty.[/yellow]"
            )

    n_frames = len(probs)
    frame_duration_s = frame_shift_ms / 1000.0
    end_time_s = n_frames * frame_duration_s
    end_frame = n_frames - 1

    sentinel_start: ValleyTrough = {
        "frame": 0,
        "global_frame": 0,
        "prob": probs[0] if n_frames > 0 else 0.0,
        "time_s": 0.0,
        "global_time_s": 0.0,
        "valley": valley_troughs[0]["valley"].copy(),
    }
    sentinel_end: ValleyTrough = {
        "frame": end_frame,
        "global_frame": end_frame,
        "prob": probs[-1] if n_frames > 0 else 0.0,
        "time_s": end_time_s,
        "global_time_s": end_time_s,
        "valley": valley_troughs[-1]["valley"].copy(),
    }

    anchors: List[ValleyTrough] = (
        [sentinel_start] + list(valley_troughs) + [sentinel_end]
    )

    segments: List[TroughToTroughSegment] = []
    audio_slices: List[np.ndarray] = []
    total_audio_samples = len(audio_np) if audio_np is not None else 0

    for idx in range(len(anchors) - 1):
        vt_start = anchors[idx]
        vt_end = anchors[idx + 1]
        is_first = idx == 0
        is_last = idx == len(anchors) - 2

        start_s: float = float(vt_start["global_time_s"])
        end_s: float = float(vt_end["global_time_s"])
        duration_s: float = round(end_s - start_s, 4)
        start_frame: int = int(vt_start["global_frame"])
        end_frame_seg: int = int(vt_end["global_frame"])

        segment_probs: Optional[List[float]] = None
        prob_stats: Optional[Dict[str, float]] = None
        assigned_segments: Optional[List[SpeechSegment]] = None

        if with_scores:
            segment_probs_slice = probs[start_frame : end_frame_seg + 1]
            segment_probs = segment_probs_slice
            if segment_probs_slice:
                prob_stats = {
                    "mean": float(np.mean(segment_probs_slice)),
                    "max": float(np.max(segment_probs_slice)),
                    "min": float(np.min(segment_probs_slice)),
                    "std": float(np.std(segment_probs_slice)),
                    "median": float(np.median(segment_probs_slice)),
                    "num_frames": len(segment_probs_slice),
                }
                console.print(
                    f"[blue]Segment {idx}: probs stats - "
                    f"mean={prob_stats['mean']:.4f}, max={prob_stats['max']:.4f}, "
                    f"min={prob_stats['min']:.4f}, std={prob_stats['std']:.4f}, "
                    f"median={prob_stats['median']:.4f}, "
                    f"frames={prob_stats['num_frames']}[/blue]"
                )

            # Assign speech segments that fall within this trough-to-trough segment's time range
            # Now both start/end are in seconds (thanks to return_seconds=True above)
            if all_speech_segments:
                assigned_segments = [
                    seg
                    for seg in all_speech_segments
                    if seg["start"] <= end_s and seg["end"] >= start_s
                ]
                console.print(
                    f"[blue]Segment {idx} [{start_s:.3f}s - {end_s:.3f}s]: "
                    f"assigned {len(assigned_segments)} speech segments[/blue]"
                )

        segment: TroughToTroughSegment = {
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": duration_s,
            "start_frame": start_frame,
            "end_frame": end_frame_seg,
            "trough_start": None if is_first else dict(vt_start),
            "trough_end": None if is_last else dict(vt_end),
            "segment_probs": segment_probs if with_scores else None,
            "prob_stats": prob_stats if with_scores else None,
            "segments": assigned_segments if with_scores else None,
        }
        segments.append(segment)

        if with_audio and audio_np is not None:
            start_sample = int(start_s * sample_rate)
            end_sample = int(end_s * sample_rate)
            start_sample = max(0, start_sample)
            end_sample = min(total_audio_samples, end_sample)
            audio_slice = audio_np[start_sample:end_sample]
            audio_slices.append(audio_slice)
            console.print(
                f"[magenta]Segment {idx}: [{start_s:.3f}s - {end_s:.3f}s] "
                f"→ audio samples [{start_sample}:{end_sample}] "
                f"({len(audio_slice)} samples, {len(audio_slice) / sample_rate:.3f}s)[/magenta]"
            )

    console.print(
        f"[green]extract_trough_to_trough: Created {len(segments)} segments "
        f"from {len(valley_troughs)} trough(s)"
        + (" with audio slices" if with_audio else "")
        + (" with probability scores" if with_scores else "")
        + (" with speech segments" if with_scores and all_speech_segments else "")
        + ".[/green]"
    )

    if with_audio:
        segments_with_audio = list(zip(segments, audio_slices))
        if with_scores:
            return segments_with_audio, probs
        return segments_with_audio

    if with_scores:
        return segments, probs

    return segments


def smooth_vad_probs(probs: List[float], window: int = 20) -> List[float]:
    """Light moving average smoothing to reduce jitter in VAD probabilities."""
    if window <= 1 or len(probs) <= window:
        return probs[:]
    x = np.array(probs, dtype=float)
    smoothed = np.convolve(x, np.ones(window) / window, mode="same")
    smoothed[0] = (x[0] + x[1]) / 2 if len(x) > 1 else x[0]
    if len(x) > 2:
        smoothed[-1] = (x[-1] + x[-2]) / 2
    return smoothed.tolist()


def compute_valley_score(
    min_prob: float,
    mean_prob: float,
    duration_s: float,
    max_duration_ref: float = 1.0,
    w_depth: float = 0.4,
    w_mean: float = 0.4,
    w_duration: float = 0.2,
) -> float:
    """
    Composite score for valley quality. Higher score = stronger silence (safe to cut).

    Args:
        min_prob: Minimum probability in valley
        mean_prob: Mean probability in valley
        duration_s: Duration in seconds
        max_duration_ref: Duration normalization cap
        w_depth, w_mean, w_duration: Weights

    Returns:
        float score in [0, 1]
    """
    duration_norm = min(duration_s / max_duration_ref, 1.0)
    score = (
        w_depth * (1.0 - min_prob)
        + w_mean * (1.0 - mean_prob)
        + w_duration * duration_norm
    )
    return float(score)


def compute_trough_score(
    min_prob: float,
    prominence: float,
    width: float,
    max_width_ref: float = 20.0,
    w_depth: float = 0.4,
    w_prominence: float = 0.4,
    w_width: float = 0.2,
) -> float:
    """Score how safe a trough is for cutting. Higher score = safer cut point."""
    depth_score = 1.0 - min_prob
    prominence_norm = min(prominence / 0.5, 1.0) if prominence is not None else 0.0
    width_norm = min(width / max_width_ref, 1.0) if width is not None else 0.0
    score = (
        w_depth * depth_score + w_prominence * prominence_norm + w_width * width_norm
    )
    return float(score)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
