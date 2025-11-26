from __future__ import annotations
import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline
from tqdm.auto import tqdm
from typing import BinaryIO, List, Tuple, Union
from faster_whisper.transcribe import Segment, TranscriptionInfo
from pathlib import Path
from typing import Generator, Optional
from faster_whisper.audio import decode_audio

AudioInput = Union[str, BinaryIO, np.ndarray]

def _get_default_settings(**overrides) -> dict:
    """Shared transcription defaults – easily overridable."""
    return {
        "beam_size": 5,
        "temperature": [0.0, 0.2, 0.4],
        "repetition_penalty": 1.1,
        "condition_on_previous_text": True,
        "word_timestamps": True,
        "vad_filter": True,
        "language": "ja",
        "task": "translate",                  # Japanese → English
        **overrides,
    }

def transcribe_audio2(
    audio: AudioInput,
    model: WhisperModel,
    show_progress: bool = True,
    **kwargs,
) -> Tuple[List[Segment], TranscriptionInfo]:
    """
    Transcribe a single audio with optional tqdm progress bar.
    """
    settings = _get_default_settings(**kwargs)

    segments_iter, info = model.transcribe(audio, **settings)

    segments: List[Segment] = []
    iterator = tqdm(
        segments_iter,
        desc="Transcribing",
        unit="segment",
        leave=False,
        disable=not show_progress,
    ) if show_progress else segments_iter

    for segment in iterator:
        segments.append(segment)

    return segments, info


def transcribe_batch_audio(
    audios: List[AudioInput],
    model: WhisperModel,
    batch_size: int = 16,
    show_progress: bool = True,
    progress_desc: str = "Batch transcribing",
    **kwargs,
) -> List[Tuple[List[Segment], TranscriptionInfo]]:
    """
    Transcribe multiple audios with beautiful nested tqdm progress bars:
      - Outer bar: files
      - Inner bar: segments per file (only when using BatchedInferencePipeline)
    """
    if not audios:
        return []

    settings = _get_default_settings(log_progress=False, **kwargs)
    batched_model = BatchedInferencePipeline(model=model)

    results: List[Tuple[List[Segment], TranscriptionInfo]] = []

    file_iter = tqdm(
        enumerate(audios),
        total=len(audios),
        desc=progress_desc,
        unit="file",
        leave=True,
        disable=not show_progress,
    )

    for idx, audio in file_iter:
        file_iter.set_postfix_str(f"File {idx + 1}/{len(audios)}", refresh=True)

        segments_iter, info = batched_model.transcribe(
            audio,
            batch_size=batch_size,
            **settings,
        )

        segments: List[Segment] = []
        seg_iter = tqdm(
            segments_iter,
            desc="  Segments",
            unit="seg",
            leave=False,
            disable=not show_progress,
        ) if show_progress else segments_iter

        for segment in seg_iter:
            segments.append(segment)

        results.append((segments, info))

    return results

AudioInput = Union[str, Path, np.ndarray]

# Yield type: one result per segment as soon as its chunk is done
TranscriptionStream = Generator[tuple[Segment, TranscriptionInfo, int], None, None]

def transcribe_audio(
    audio_input: AudioInput,
    model_name: str = "large-v3",
    device: str = "auto",
    compute_type: str = "int8_float32",
    overlap_seconds: float = 8.0,
    chunk_length_seconds: int = 30,
    vad_filter: bool = True,
    word_timestamps: bool = True,
    hotwords: Optional[List[str]] = None,
    **transcribe_kwargs,
) -> TranscriptionStream:
    """
    Streaming transcription of long audio with correct overlap handling.

    Yields:
        (Segment, TranscriptionInfo, int) – one tuple per segment:
            - adjusted Segment with correct global timestamps
            - TranscriptionInfo from the current chunk
            - chunk_idx (0-based index of the chunk this segment came from)
    """
    # Accept str, Path, or already-loaded np.ndarray
    if isinstance(audio_input, (str, Path)):
        audio_path = Path(audio_input)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_source = str(audio_path)
    else:
        audio_source = audio_input  # already np.ndarray

    print(f"Loading model {model_name} on {device} ({compute_type})...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    print("Loading and resampling audio to 16kHz...")
    audio = decode_audio(audio_source, sampling_rate=16000) if isinstance(audio_source, (str, Path)) else audio_source
    sample_rate = 16000.0

    chunk_samples = int(chunk_length_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)

    print(f"Starting streaming transcription (overlap={overlap_seconds}s, chunk={chunk_length_seconds}s)...\n")

    start_sample = 0
    chunk_idx = 0

    while start_sample < len(audio):
        end_sample = start_sample + chunk_samples
        chunk_audio = audio[start_sample:end_sample]

        chunk_start_sec = start_sample / sample_rate
        chunk_end_sec = min(end_sample, len(audio)) / sample_rate
        print(f"Transcribing chunk {chunk_idx + 1} @ {chunk_start_sec:.2f}s → {chunk_end_sec:.2f}s")

        segs, chunk_info = model.transcribe(
            chunk_audio,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            hotwords=",".join(hotwords) if hotwords else None,
            **transcribe_kwargs,
        )

        offset = start_sample / sample_rate

        for seg in segs:
            # Adjust word-level timestamps
            adjusted_words = None
            if word_timestamps and seg.words:
                adjusted_words = [
                    type("Word", (), {
                        "start": (w.start + offset) if w.start is not None else None,
                        "end": (w.end + offset) if w.end is not None else None,
                        "word": w.word,
                        "probability": w.probability,
                    })()
                    for w in seg.words
                ]

            adjusted_segment = Segment(
                id=seg.id,
                seek=seg.seek,
                start=seg.start + offset,
                end=seg.end + offset,
                text=seg.text,
                tokens=seg.tokens,
                temperature=seg.temperature,
                avg_logprob=seg.avg_logprob,
                compression_ratio=seg.compression_ratio,
                no_speech_prob=seg.no_speech_prob,
                words=adjusted_words,
            )

            # Yield immediately with per-chunk info
            yield adjusted_segment, chunk_info, chunk_idx

        # Advance chunk with overlap
        start_sample += chunk_samples - overlap_samples
        chunk_idx += 1

    # Optional: yield a final None + info if you want a clean end signal
    # yield None, chunk_info  # uncomment if needed downstream
