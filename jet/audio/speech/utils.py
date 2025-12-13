import numpy as np
import json
from typing import Dict, Any
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


def save_completed_segment(
    segment_root: str | Path,
    counter: int,
    ts: Dict[str, Any],
    audio_np: np.ndarray,
    is_final_utterance: bool = False,
) -> int:
    """
    Save a single completed speech segment to disk with WAV and metadata.json.
    
    Returns the updated counter (incremented by 1).
    """
    segment_root = Path(segment_root)
    start_sample = int(ts["start"])
    end_sample = len(audio_np) if is_final_utterance else int(ts["end"])
    
    seg_audio = audio_np[start_sample:end_sample]
    counter += 1
    seg_dir = segment_root / f"segment_{counter:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    
    wav_path = seg_dir / "sound.wav"
    save_wav_file(wav_path, seg_audio)
    
    meta: Dict[str, Any] = {
        "segment_id": counter,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "start_sec": round(start_sample / SAMPLE_RATE, 3),
        "end_sec": round(end_sample / SAMPLE_RATE, 3),
        "duration_sec": round((end_sample - start_sample) / SAMPLE_RATE, 3),
        "recorded_at": datetime.utcnow().isoformat() + "Z",
        "source": "record_from_mic_speech_detection",
    }
    if is_final_utterance:
        meta["note"] = "final_utterance"
        meta["status"] = "completed_partial"  # last segment may be cut off
    else:
        meta["status"] = "completed"
    
    (seg_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"Saved {'final utterance' if is_final_utterance else 'complete segment'} → {seg_dir.name}")
    
    return counter

def display_segments(speech_ts):
    # Derive total recorded seconds as the end of the last speech segment, or 0 if none
    if speech_ts:
        recorded_seconds = max(seg["end"] for seg in speech_ts)
    else:
        recorded_seconds = 0.0

    table = Table(title=f"Speech segments (total {recorded_seconds:.1f}s recorded)")
    table.add_column("Segment", style="cyan")
    table.add_column("Start (s)", justify="right")
    table.add_column("End (s)", justify="right")
    table.add_column("Duration (s)", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Status", style="green")

    for i, seg in enumerate(speech_ts, 1):
        table.add_row(
            str(seg['idx'] + 1),
            f"{seg['start']:.2f}",
            f"{seg['end']:.2f}",
            f"{seg['duration']:.2f}",
            f"{seg['prob']:.2f}",
            "active",  # all currently detected segments are still active
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
