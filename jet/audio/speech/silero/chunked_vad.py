"""
Silero VAD Fixed-Duration Chunk Processor

This module provides utilities for processing audio with Silero VAD to obtain
speech confidence probabilities over fixed-length time chunks (e.g., every 0.25 seconds).

Key features:
- Returns typed results with start/end timestamps and aggregated speech probability.
- Supports non-overlapping or overlapping chunks.
- Works with both ONNX and JIT versions of the Silero VAD model.
- Configurable aggregation: mean (default), max, or median.
- Handles edge cases like short audio or incomplete final chunks gracefully.

Intended to complement the original `get_speech_timestamps` and `VADIterator`
by providing regular, fixed-interval probability output suitable for visualization,
feature extraction, or downstream ML models.
"""

from typing import List, TypedDict, Literal
import torch


class SpeechProbChunk(TypedDict):
    """Typed dictionary for fixed-duration speech probability chunks"""
    start_sec: float          # start time in seconds
    end_sec: float            # end time in seconds
    duration_sec: float       # = end_sec - start_sec
    speech_prob: float        # average speech probability in [0.0, 1.0]
    num_windows: int          # how many VAD windows contributed to this chunk


def get_speech_probabilities_chunks(
    audio: torch.Tensor,
    model,
    sampling_rate: int = 16000,
    chunk_seconds: float = 0.5,
    overlap_seconds: float = 0.25,
    min_chunk_samples: int | None = None,
    aggregation: Literal["mean", "max", "median"] = "mean",
) -> List[SpeechProbChunk]:
    """
    Split audio into fixed-duration chunks and return speech confidence for each.

    Returns
    -------
    List[SpeechProbChunk]
        One typed dict per chunk with timestamps and speech probability.
    """
    if audio.dim() != 1:
        raise ValueError("Audio must be 1D tensor")
    if sampling_rate not in {8000, 16000}:
        raise ValueError("Only 8000 and 16000 Hz are supported by Silero VAD")

    chunk_samples = round(chunk_seconds * sampling_rate)
    step_samples = round((chunk_seconds - overlap_seconds) * sampling_rate)

    if chunk_samples <= 0 or step_samples <= 0:
        raise ValueError("chunk_seconds and step must be > 0")

    window_size = 512 if sampling_rate == 16000 else 256

    # Ensure we can run the model (pad if audio is shorter than one window)
    if len(audio) < window_size:
        audio = torch.nn.functional.pad(audio, (0, window_size - len(audio)))

    # Run full forward pass — works for both ONNX and JIT models
    if hasattr(model, "audio_forward"):
        probs_tensor = model.audio_forward(audio.unsqueeze(0), sampling_rate)  # [1, num_windows]
    else:
        # Manual chunking for pure JIT models
        probs = []
        model.reset_states()
        for i in range(0, len(audio), window_size):
            chunk = audio[i:i + window_size]
            if len(chunk) < window_size:
                chunk = torch.nn.functional.pad(chunk, (0, window_size - len(chunk)))
            prob = model(chunk.unsqueeze(0), sampling_rate).item()
            probs.append(prob)
        probs_tensor = torch.tensor(probs).unsqueeze(0)

    probs = probs_tensor.squeeze(0).tolist()           # List[float]
    hop_sec = window_size / sampling_rate

    results: List[SpeechProbChunk] = []
    pos_samples = 0
    total_samples = len(audio)

    while pos_samples + chunk_samples <= total_samples:
        end_samples = pos_samples + chunk_samples

        # Window indices that fall into this chunk
        start_idx = pos_samples // window_size
        end_idx = end_samples // window_size          # exclusive

        chunk_probs = probs[start_idx:end_idx]
        if not chunk_probs:
            break

        # Aggregation
        if aggregation == "mean":
            speech_prob = sum(chunk_probs) / len(chunk_probs)
        elif aggregation == "max":
            speech_prob = max(chunk_probs)
        else:  # median
            sorted_probs = sorted(chunk_probs)
            mid = len(sorted_probs) // 2
            speech_prob = sorted_probs[mid] if len(sorted_probs) % 2 else sum(sorted_probs[mid - 1:mid + 1]) / 2

        results.append(SpeechProbChunk(
            start_sec=round(pos_samples / sampling_rate, 4),
            end_sec=round(end_samples / sampling_rate, 4),
            duration_sec=round(chunk_seconds, 4),
            speech_prob=round(float(speech_prob), 4),
            num_windows=len(chunk_probs),
        ))

        pos_samples += step_samples

    # Optional: last incomplete chunk
    if min_chunk_samples is not None and pos_samples < total_samples:
        remaining = total_samples - pos_samples
        if remaining >= min_chunk_samples:
            start_idx = pos_samples // window_size
            chunk_probs = probs[start_idx:]
            if chunk_probs:
                speech_prob = sum(chunk_probs) / len(chunk_probs) if aggregation == "mean" else (
                    max(chunk_probs) if aggregation == "max" else float(sorted(chunk_probs)[len(chunk_probs)//2])
                )
                results.append(SpeechProbChunk(
                    start_sec=round(pos_samples / sampling_rate, 4),
                    end_sec=round(total_samples / sampling_rate, 4),
                    duration_sec=round(remaining / sampling_rate, 4),
                    speech_prob=round(speech_prob, 4),
                    num_windows=len(chunk_probs),
                ))

    return results