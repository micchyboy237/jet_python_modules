# transcription.py
import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline
from tqdm.auto import tqdm
from typing import BinaryIO, List, Tuple, Union

from faster_whisper.transcribe import Segment, TranscriptionInfo

AudioInput = Union[str, BinaryIO, np.ndarray]


def _get_default_settings(**overrides) -> dict:
    """Shared transcription defaults – easily overridable."""
    return {
        "beam_size": 1,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "vad_filter": True,
        **overrides,
    }


def transcribe_audio(
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
