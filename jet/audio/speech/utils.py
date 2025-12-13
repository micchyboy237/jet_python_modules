import numpy as np
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import torch
import wave
from rich.table import Table

from jet.logger import logger
from jet.audio.helpers.silence import (
    SAMPLE_RATE,
    DTYPE,
    CHANNELS,
)


# jet_python_modules/jet/audio/speech/utils.py
def save_completed_segment(
    segment_root: str | Path,
    counter: int,
    ts: Dict[str, Any],
    audio_np: np.ndarray,
    *,
    trim_silence: bool = False,
    silence_threshold: Optional[float] = None,
) -> int:
    """
    Save a single completed speech segment to disk with WAV and metadata.json.

    Args:
        segment_root: Directory where segment folders will be created.
        counter: Current segment counter (will be incremented and returned).
        ts: Timestamp dictionary from Silero VAD containing "start" and "end" in samples.
        audio_np: Full recorded audio (untrimmed) as np.ndarray.
        trim_silence: If True, removes leading/trailing silence from this segment only.
        silence_threshold: Silence RMS threshold for trimming (required if trim_silence=True).

    Returns:
        Updated counter (counter + 1).
    """
    from jet.audio.helpers.silence import trim_silent_chunks, calibrate_silence_threshold

    segment_root = Path(segment_root)
    start_sample = int(ts["start"])
    end_sample = int(ts["end"])

    # Extract raw segment (including possible silence at edges)
    seg_audio = audio_np[start_sample:end_sample]

    # Optionally trim silence only within this segment
    if trim_silence:
        if silence_threshold is None:
            silence_threshold = calibrate_silence_threshold()
        # Convert single ndarray to list of chunks for existing trim function
        chunk_size = int(0.1 * SAMPLE_RATE)  # 100 ms chunks – reasonable for trimming
        chunks = [seg_audio[i:i + chunk_size] for i in range(0, len(seg_audio), chunk_size)]
        trimmed_chunks = trim_silent_chunks(chunks, silence_threshold)
        if not trimmed_chunks:
            # Extremely rare – entire segment was silence
            seg_audio = np.array([], dtype=seg_audio.dtype)
        else:
            seg_audio = np.concatenate(trimmed_chunks, axis=0)
        # Adjust timestamps to reflect trimmed audio
        trimmed_samples_removed_front = len(chunks[0]) if chunks and trimmed_chunks and len(trimmed_chunks) > 0 else 0
        # More precise: calculate actual samples removed from start
        total_original = len(seg_audio) + (len(chunks) - len(trimmed_chunks)) * chunk_size
        # Simpler & sufficient: recalculate start/end relative to trimmed
        start_sample = start_sample + (len(seg_audio) - len(seg_audio))  # placeholder; we update meta later
    counter += 1
    seg_dir = segment_root / f"segment_{counter:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    wav_path = seg_dir / "sound.wav"
    save_wav_file(wav_path, seg_audio)

    # Metadata reflects the actual saved (possibly trimmed) audio
    actual_start_sample = start_sample
    actual_end_sample = start_sample + len(seg_audio)
    meta: Dict[str, Any] = {
        "segment_id": counter,
        "original_start_sample": int(ts["start"]),
        "original_end_sample": int(ts["end"]),
        "saved_start_sample": actual_start_sample,
        "saved_end_sample": actual_end_sample,
        "start_sec": round(actual_start_sample / SAMPLE_RATE, 3),
        "end_sec": round(actual_end_sample / SAMPLE_RATE, 3),
        "duration_sec": round(len(seg_audio) / SAMPLE_RATE, 3),
        "recorded_at": datetime.utcnow().isoformat() + "Z",
        "source": "record_from_mic_speech_detection",
        "status": "completed",
        "trimmed": trim_silence,
    }
    if trim_silence:
        meta["silence_threshold_used"] = silence_threshold

    (seg_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"Saved complete segment → {seg_dir.name}")
    return counter

def display_segments(speech_ts):
    """Display detected speech segments in a clean Rich table with correct time in seconds."""
    if not speech_ts:
        return

    # Total recorded time approximated by the end of the last speech segment (in samples)
    total_samples = max(seg["end"] for seg in speech_ts)
    recorded_seconds = total_samples / SAMPLE_RATE

    table = Table(title=f"Speech segments (total ~{recorded_seconds:.1f}s recorded)")

    table.add_column("Segment", style="cyan", justify="right")
    table.add_column("Start (s)", justify="right")
    table.add_column("End (s)", justify="right")
    table.add_column("Duration (s)", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Status", style="green")

    for i, seg in enumerate(speech_ts, 1):
        start_sec = seg["start"] / SAMPLE_RATE
        end_sec = seg["end"] / SAMPLE_RATE
        duration_sec = (seg["end"] - seg["start"]) / SAMPLE_RATE

        table.add_row(
            str(i),
            f"{start_sec:.2f}",
            f"{end_sec:.2f}",
            f"{duration_sec:.2f}",
            f"{seg.get('prob', seg.get('score', '-')):.2f}",
            "active" if i == len(speech_ts) else "",
        )

    from rich import print as rprint
    rprint("\n", table, "\n")


def save_wav_file(filename, audio_data: np.ndarray):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(filename), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(np.dtype(DTYPE).itemsize)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    logger.info(f"Audio saved to {filename}")


def convert_audio_to_tensor(audio_data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy audio array or list of chunks to torch tensor suitable for Silero VAD.
    - Ensures mono
    - Converts to float32 in range [-1.0, 1.0]
    - Requires 16kHz input!
    """
    # Accept either a single np.ndarray or a list of chunks
    if isinstance(audio_data, list):
        audio = np.concatenate(audio_data, axis=0)
    else:
        audio = np.asarray(audio_data)

    # Normalize integer PCM to float32 in [-1, 1]
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    elif audio.dtype == np.float64:
        audio = audio.astype(np.float32)
    # If already float, ensure [-1, 1]
    elif np.issubdtype(audio.dtype, np.floating):
        audio = np.clip(audio, -1.0, 1.0)
    else:
        raise ValueError("Unsupported audio dtype")

    tensor = torch.from_numpy(audio)

    # Convert to mono if multi-channel (average channels)
    if tensor.ndim > 1:
        tensor = tensor.mean(dim=1)

    # Sanity checks
    assert tensor.abs().max() <= 1.0 + 1e-5, "Audio not normalized!"
    assert SAMPLE_RATE == 16000, "Wrong sample rate for Silero VAD: must be 16000 Hz"

    return tensor  # shape: (N_samples,), float32, [-1, 1], 16kHz


class SpeechSegmentTracker:
    def __init__(self):
        self.speech_ts = []
        self.curr_segment = None

    def update_speech_ts(self, speech_ts):
        start_sec = speech_ts[-1]["start"]
        self.speech_ts = speech_ts
