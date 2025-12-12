import torch
import wave
import numpy as np
from silero_vad import get_speech_timestamps, load_silero_vad
import sounddevice as sd
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
from rich.table import Table
from collections import deque
import json
from datetime import datetime

from jet.logger import logger
from jet.audio.helpers.silence import (
    SAMPLE_RATE,
    DTYPE,
    CHANNELS,
    calibrate_silence_threshold,
    detect_silence,
    trim_silent_chunks,
)

# Ensure best results: Silero VAD is reliable only at 16kHz, mono audio
# Change the constants in jet/audio/helpers/silence.py to:
# SAMPLE_RATE = 16000, DTYPE = 'int16', CHANNELS = 1

silero_model = load_silero_vad(onnx=False)

if not hasattr(__import__(__name__), "record_from_mic_segment_state"):
    # Safe module-level, works if file re-executes (avoid function attribute crosstalk)
    record_from_mic_segment_state = {
        "finalized_up_to_sample": 0,
        "last_seen_segments": [],
        "counter": 0,
    }
else:
    record_from_mic_segment_state = getattr(__import__(__name__), "record_from_mic_segment_state")


def record_from_mic(
    duration: Optional[int] = None,
    silence_threshold: Optional[float] = None,
    silence_duration: float = 2.0,
    trim_silent: bool = True,
    *,
    output_dir: Optional[Path | str] = None,
) -> Optional[np.ndarray]:
    """Record audio from microphone with silence detection and progress tracking.

    When ``output_dir`` is supplied each detected speech segment (from Silero VAD)
    is saved under ``<output_dir>/segment/segment_<num>/`` as:
        - sound.wav
        - metadata.json

    Args:
        duration: Maximum recording duration in seconds (None = indefinite).
        silence_threshold: Silence level in RMS (None = auto-calibrated).
        silence_duration: Seconds of continuous silence to stop recording.
        trim_silent: If True, removes silent sections from the final audio.
        output_dir: Optional output directory to save segments and metadata as they are detected.
    """
    global silero_model
    silence_threshold = silence_threshold if silence_threshold is not None else calibrate_silence_threshold()

    duration_str = f"{duration}s" if duration is not None else "indefinite"
    logger.info(
        f"Starting recording: {CHANNELS} channel{'s' if CHANNELS > 1 else ''}, "
        f"max duration {duration_str}, silence threshold {silence_threshold:.6f}, "
        f"silence duration {silence_duration}s"
    )

    chunk_size = int(SAMPLE_RATE * 0.5)  # 0.5 second chunks
    max_frames = int(duration * SAMPLE_RATE) if duration is not None else float('inf')
    silence_frames = int(silence_duration * SAMPLE_RATE)
    grace_frames = int(SAMPLE_RATE * 1.0)  # 1-second grace period

    audio_data: List[np.ndarray] = []
    silent_count = 0
    recorded_frames = 0

    # Segment saving setup
    save_segments = output_dir is not None
    if save_segments:
        output_dir = Path(output_dir)
        segment_root = output_dir / "segment"
        segment_root.mkdir(parents=True, exist_ok=True)
    # Use module/global record_from_mic_segment_state as state
    state = record_from_mic_segment_state
    state["counter"] = 0
    state["finalized_up_to_sample"] = 0
    state["last_seen_segments"] = []

    pbar_kwargs = {'total': duration, 'desc': "Recording", 'unit': "s",
                   'leave': True} if duration is not None else {'desc': "Recording", 'unit': "s", 'leave': True}
    with tqdm(**pbar_kwargs) as pbar:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE
        )
        with stream:
            while recorded_frames < max_frames:
                chunk = stream.read(chunk_size)[0]
                audio_data.append(chunk)
                recorded_frames += chunk_size
                pbar.update(0.5)

                # Skip silence detection during grace period
                if recorded_frames > grace_frames and detect_silence(chunk, silence_threshold):
                    silent_count += chunk_size
                    if silent_count >= silence_frames:
                        logger.info(
                            f"Silence detected for {silence_duration}s, stopping recording")
                        break
                else:
                    silent_count = 0

                # Run Silero VAD every few chunks to detect speech segments in real-time
                if len(audio_data) % 5 == 0:
                    trimmed_acc = trim_silent_chunks(audio_data, silence_threshold)
                    if trimmed_acc:
                        audio_tensor = convert_audio_to_tensor(trimmed_acc)
                        speech_ts = get_speech_timestamps(audio=audio_tensor, model=silero_model)
                        display_segments(speech_ts)

                        if save_segments and speech_ts:
                            full_audio_np = np.concatenate(audio_data, axis=0)

                            current_max_end = max(ts["end"] for ts in speech_ts)
                            newly_finalized = []

                            # Segments whose endpoint is now <= finalized_up_to_sample AND not in prev last_seen_segments
                            for ts in speech_ts:
                                if ts["end"] <= state["finalized_up_to_sample"]:
                                    if not any(abs(ts["start"] - s["start"]) < 100 and abs(ts["end"] - s["end"]) < 100
                                               for s in state["last_seen_segments"]):
                                        newly_finalized.append(ts)

                            # Also, if any segment from old state["last_seen_segments"] disappeared (not in this VAD result): finalize those
                            prev_set = {(s["start"], s["end"]) for s in state["last_seen_segments"]}
                            curr_set = {(s["start"], s["end"]) for s in speech_ts}
                            for old in state["last_seen_segments"]:
                                if (old["start"], old["end"]) not in curr_set:
                                    newly_finalized.append(old)

                            for ts in newly_finalized:
                                seg_audio = full_audio_np[ts["start"]:ts["end"]]
                                state["counter"] += 1
                                seg_dir = segment_root / f"segment_{state['counter']:03d}"
                                seg_dir.mkdir(parents=True, exist_ok=True)

                                wav_path = seg_dir / "sound.wav"
                                save_wav_file(wav_path, seg_audio)

                                meta = {
                                    "segment_id": state["counter"],
                                    "start_sample": int(ts["start"]),
                                    "end_sample": int(ts["end"]),
                                    "start_sec": round(ts["start"] / SAMPLE_RATE, 3),
                                    "end_sec": round(ts["end"] / SAMPLE_RATE, 3),
                                    "duration_sec": round((ts["end"] - ts["start"]) / SAMPLE_RATE, 3),
                                    "recorded_at": datetime.utcnow().isoformat() + "Z",
                                    "source": "record_from_mic_speech_detection",
                                    "finalized_up_to_sample": state["finalized_up_to_sample"],
                                }
                                (seg_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                                logger.info(f"Saved complete segment → {seg_dir.name}")

                            state["finalized_up_to_sample"] = current_max_end
                            state["last_seen_segments"] = [dict(ts) for ts in speech_ts]

    if not audio_data:
        logger.warning("No audio recorded")
        return None

    # Trim silent chunks only when requested
    if trim_silent:
        trimmed_data = trim_silent_chunks(audio_data, silence_threshold)
        if not trimmed_data:
            logger.warning("All chunks were silent after trimming")
            return None
        audio_data = np.concatenate(trimmed_data, axis=0)
    else:
        audio_data = np.concatenate(audio_data, axis=0)

    # === FINAL FLUSH: save any remaining speech ===
    if save_segments and state["last_seen_segments"]:
        full_audio_np = np.concatenate(audio_data, axis=0) if isinstance(audio_data, list) else audio_data
        for ts in state["last_seen_segments"]:
            # Only keep "remaining" non-finalized ones
            if ts["end"] > state["finalized_up_to_sample"]:
                start_s = ts["start"]
                end_s = len(full_audio_np)
                # Only save if at least 0.1 second
                if end_s - start_s > 0.1 * SAMPLE_RATE:
                    seg_audio = full_audio_np[start_s:end_s]
                    state["counter"] += 1
                    seg_dir = segment_root / f"segment_{state['counter']:03d}"
                    seg_dir.mkdir(parents=True, exist_ok=True)
                    save_wav_file(seg_dir / "sound.wav", seg_audio)
                    meta = {
                        "segment_id": state["counter"],
                        "start_sample": int(start_s),
                        "end_sample": int(end_s),
                        "start_sec": round(start_s / SAMPLE_RATE, 3),
                        "end_sec": round(end_s / SAMPLE_RATE, 3),
                        "duration_sec": round((end_s - start_s) / SAMPLE_RATE, 3),
                        "recorded_at": datetime.utcnow().isoformat() + "Z",
                        "source": "record_from_mic_speech_detection",
                        "note": "final_utterance"
                    }
                    (seg_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                    logger.info(f"Saved final utterance → {seg_dir.name}")

    actual_duration = len(audio_data) / SAMPLE_RATE
    logger.info(f"Recording complete, actual duration: {actual_duration:.2f}s")
    return audio_data


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


def display_segments(speech_ts):
    # Convert to seconds and print with rich table
    segments_sec = [
        {"start": t["start"] / SAMPLE_RATE, "end": t["end"] / SAMPLE_RATE}
        for t in speech_ts
    ]

    # Derive total recorded seconds as the end of the last speech segment, or 0 if none
    if segments_sec:
        recorded_seconds = max(seg["end"] for seg in segments_sec)
    else:
        recorded_seconds = 0.0

    table = Table(title=f"Speech segments (total {recorded_seconds:.1f}s recorded)")
    table.add_column("Segment", style="cyan")
    table.add_column("Start (s)", justify="right")
    table.add_column("End (s)", justify="right")
    table.add_column("Duration (s)", justify="right")

    for i, seg in enumerate(segments_sec, 1):
        table.add_row(
            str(i),
            f"{seg['start']:.2f}",
            f"{seg['end']:.2f}",
            f"{seg['end'] - seg['start']:.2f}",
        )

    from rich import print as rprint
    rprint("\n", table, "\n")
