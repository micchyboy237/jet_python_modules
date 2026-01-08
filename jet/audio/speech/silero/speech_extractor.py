# speech_extractor.py

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Literal, Optional, Union
from typing import TypedDict

from silero_vad import read_audio, load_silero_vad
import matplotlib.pyplot as plt

import shutil
import itertools
import torch
import numpy as np

from jet.file.utils import save_file
from jet.logger import logger

silero_model = load_silero_vad(onnx=False)

AudioInput = Union[str, Path, np.ndarray, torch.Tensor, bytes]
Unit = Literal['ms', 'seconds']


@dataclass
class SpeechSegment:
    num: int
    start_s: float  # start time in seconds (3 decimals)
    end_s: float    # end time in seconds (3 decimals)
    duration_s: float  # duration in seconds (3 decimals)
    frame_start: int    # index of first VAD frame in the segment
    frame_end: int      # index of last VAD frame in the segment (exclusive)
    frame_count: int    # number of VAD frames in the segment
    stats: "SegmentStats"

    def to_dict(self, *, timing_unit: Unit = 'ms') -> dict:
        """Return a dict representation with timing fields converted to seconds (3 decimals)."""
        data = asdict(self)
        # Fields are already in seconds; round timing fields to 3 decimals for output
        for field in ('start_s', 'end_s', 'duration_s'):
            if data[field] is not None:
                data[field] = round(data[field], 3)
        # Frame fields are integers → no rounding needed
        return data


class SegmentStats(TypedDict):
    avg_prob: float
    min_prob: float
    max_prob: float
    std_prob: float
    pct_above_threshold: float
    first_prob: float     # probability of the first frame in the segment
    last_prob: float      # probability of the last frame in the segment

class VadResults(TypedDict):
    probs: list[float]  # serialized for JSON compatibility
    segments: List[dict]  # each segment already converted via to_dict()
    meta: dict


def extract_segments(
    audio: Optional[AudioInput] = None,
    probs: Optional[np.ndarray] = None,
    threshold: float = 0.3,
    low_threshold: float = 0.1,
    sampling_rate: int = 16000,
) -> List[SpeechSegment]:
    """Create natural raw speech segments with low-probability boundaries."""

    if audio is not None and probs is None:
        probs = extract_probs(audio)
    elif probs is None or len(probs) == 0:
        raise ValueError("Either `audio` must be provided, or `probs` must not be None or empty.")

    raw_segments = []
    # Step 1: Find core speech regions using a higher threshold
    core_threshold = max(threshold, 0.6)  # at least 0.6, or use main threshold if higher
    is_core = probs > core_threshold

    window_size_samples = 512 if sampling_rate == 16000 else 256

    step_sec = window_size_samples / sampling_rate

    # Step 2: Group contiguous core regions
    for group_key, group in itertools.groupby(enumerate(is_core), key=lambda x: x[1]):
        if not group_key:
            continue
        indices = [i for i, _ in group]
        if not indices:
            continue
        core_start = indices[0]
        core_end = indices[-1] + 1  # exclusive

        # Step 3: Expand backward until low prob or silence
        expand_start = core_start
        while expand_start > 0 and probs[expand_start - 1] > low_threshold:
            expand_start -= 1

        # Step 4: Expand forward
        expand_end = core_end
        max_frames = len(probs)
        while expand_end < max_frames and probs[expand_end] > low_threshold:
            expand_end += 1

        # Now the segment starts/ends near or in low-probability zones
        raw_probs = probs[expand_start:expand_end]
        if len(raw_probs) == 0:
            continue

        # Compute timings in seconds with 3-decimal precision
        start_s = round(expand_start * step_sec, 3)
        end_s = round(expand_end * step_sec, 3)
        duration_s = round(end_s - start_s, 3)

        raw_seg = SpeechSegment(
            num=len(raw_segments) + 1,
            start_s=start_s,
            end_s=end_s,
            duration_s=duration_s,
            frame_start=expand_start,
            frame_end=expand_end,
            frame_count=expand_end - expand_start,
            stats=SegmentStats(
                avg_prob=round(float(raw_probs.mean()), 3),
                min_prob=round(float(raw_probs.min()), 3),
                max_prob=round(float(raw_probs.max()), 3),
                std_prob=round(float(raw_probs.std()), 3),
                pct_above_threshold=round(
                    float(np.sum(raw_probs > threshold)) / len(raw_probs) * 100, 1
                ),
                first_prob=round(float(probs[expand_start]), 3),
                last_prob=round(float(probs[expand_end - 1]), 3),
            ),
        )

        if segment_passes_filters(raw_seg):
            raw_segments.append(raw_seg)

    return raw_segments


def extract_meta(probs: np.ndarray, sampling_rate: int = 16000) -> dict:
    """Extract metadata about the VAD processing from the probability array."""
    window_size_samples = 512 if sampling_rate == 16000 else 256
    frame_duration_s = window_size_samples / sampling_rate

    total_duration_s = round(len(probs) * frame_duration_s, 3)

    return {
        "sampling_rate": sampling_rate,
        "window_size_samples": window_size_samples,
        "frame_duration_s": round(frame_duration_s, 6),
        "total_frames": len(probs),
        "total_duration_s": total_duration_s,
    }


def save_probs_plot(
    probs: np.ndarray,
    sampling_rate: int = 16000,
    segments: Optional[List[SpeechSegment]] = None,
    output_path: Union[str, Path] = "probs_plot.png",
) -> None:
    """Save a visualization of VAD probabilities over time with optional segment highlights."""
    window_size_samples = 512 if sampling_rate == 16000 else 256
    frame_duration_s = window_size_samples / sampling_rate
    times = np.arange(len(probs)) * frame_duration_s
    output_path = str(output_path)

    plt.figure(figsize=(14, 5))
    plt.plot(times, probs, label="Speech Probability", color="steelblue")
    plt.axhline(y=0.3, color="orange", linestyle="--", label="Threshold (0.3)")
    plt.axhline(y=0.6, color="red", linestyle="--", label="Core Threshold (>=0.6)")

    if segments:
        for seg in segments:
            plt.axvspan(seg.start_s, seg.end_s, alpha=0.3, color="green", label="Speech Segment" if seg.num == 1 else "")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Speech Probability")
    plt.title("Silero VAD Probability Timeline")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.log("\nSaved speech probability plot to: ", output_path, colors=["SUCCESS", "BRIGHT_SUCCESS"])


def segment_passes_filters(
    seg: SpeechSegment,
    min_duration_s: float | None = None,  # minimum raw segment duration in seconds
    min_std_prob: float | None = None,
    min_pct_threshold: float | None = None,
) -> bool:
    """Centralized filter logic – reusable across extraction and saving."""
    if min_duration_s is not None and seg.duration_s < min_duration_s:
        return False
    if min_std_prob is not None and seg.stats["std_prob"] < min_std_prob:
        return False
    if min_pct_threshold is not None and seg.stats["pct_above_threshold"] < min_pct_threshold:
        return False
    return True


def extract_probs(audio: AudioInput, sampling_rate: int = 16000) -> np.ndarray:
    """Extract raw speech probabilities for the entire audio waveform."""
    wav = read_audio(str(audio), sampling_rate=sampling_rate).float()

    window_size_samples = 512 if sampling_rate == 16000 else 256

    probs = []
    for i in range(0, len(wav), window_size_samples):
        chunk = wav[i : i + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - len(chunk)))
        prob = silero_model(chunk.unsqueeze(0), sampling_rate).item()
        probs.append(prob)

    probs = np.array(probs)

    return probs


def process_audio(
    audio: AudioInput,
    sampling_rate: int = 16000,
    threshold: float = 0.3,
    low_threshold: float = 0.1,
) -> VadResults:
    """Full computation pipeline: extract probabilities, segments, and metadata.

    Does NOT perform any file I/O. Returns structured results for further use or saving."""
    probs = extract_probs(audio, sampling_rate=sampling_rate)
    segments = extract_segments(
        probs=probs,
        threshold=threshold,
        low_threshold=low_threshold,
        sampling_rate=sampling_rate,
    )
    meta = extract_meta(probs, sampling_rate=sampling_rate)

    return VadResults(
        probs=probs.tolist(),
        segments=[seg.to_dict() for seg in segments],
        meta=meta,
    )


if __name__ == "__main__":
    audio_file = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/results/full_recording.wav")

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = process_audio(
        audio=audio_file,
        sampling_rate=16000,
        threshold=0.3,
        low_threshold=0.1,
    )

    # Save results
    save_file(results["probs"], OUTPUT_DIR / "probs.json")
    save_file(results["segments"], OUTPUT_DIR / "segments.json")
    save_file(results["meta"], OUTPUT_DIR / "meta.json")

    # Generate and save plot
    save_probs_plot(
        probs=np.array(results["probs"]),
        sampling_rate=16000,
        segments=[SpeechSegment(**seg) for seg in results["segments"]],
        output_path=OUTPUT_DIR / "probs_plot.png",
    )