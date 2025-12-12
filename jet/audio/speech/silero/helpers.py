# jet_python_modules/jet/audio/helpers/silero_vad.py
from typing import List, Dict, Optional, Callable, Generator
from collections import deque
import torch
import numpy as np
from jet.audio.record_mic import SAMPLE_RATE  # 16000

# Load Silero VAD once (lightweight, thread-safe)
_model, _utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True,
    verbose=False,
)

(
    get_speech_timestamps,
    _,
    read_audio,
    VADIterator,
    collect_chunks,
) = _utils

SpeechSegment = Dict[str, float]          # {'start': float, 'end': float} in seconds
INTERNAL_CHUNK_SAMPLES = 512              # 32 ms @ 16 kHz – required by the model


def get_speech_timestamps_offline(
    audio: np.ndarray | torch.Tensor,
    *,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
) -> List[SpeechSegment]:
    """Offline version – returns all speech segments from a full recording."""
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio.astype(np.float32))
    if audio.ndim > 1:
        audio = audio.mean(dim=0)
    return get_speech_timestamps(
        audio,
        _model,
        threshold=threshold,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )


def silero_vad_iterator(
    *,
    threshold: float = 0.5,
    min_silence_duration_ms: int = 500,
    speech_pad_ms: int = 30,
    on_segment_complete: Optional[Callable[[SpeechSegment, np.ndarray], None]] = None,
) -> Generator[None, np.ndarray, None]:
    """
    Real-time streaming VAD iterator.
    Accepts any chunk size (0.5 s, 1 s, etc.) and calls ``on_segment_complete`` immediately
    when a speech segment ends.
    """
    vad = VADIterator(
        _model,
        threshold=threshold,
        sampling_rate=SAMPLE_RATE,
        speech_pad_ms=speech_pad_ms,
    )

    buffer: deque[np.ndarray] = deque()                    # incoming audio
    current_segment_buffer: List[np.ndarray] = []          # audio belonging to the active speech segment

    while True:
        chunk_float: Optional[np.ndarray] = yield
        if chunk_float is None:                     # end of stream → flush
            if buffer:
                vad(np.concatenate(list(buffer)), return_seconds=True)
                buffer.clear()
            final = vad(None, return_seconds=True)
            if final and current_segment_buffer:
                audio = np.concatenate(current_segment_buffer)
                if on_segment_complete:
                    on_segment_complete(final, audio)
            break

        # 1. Accumulate incoming chunk
        buffer.append(chunk_float)

        # 2. Feed the model with 512-sample pieces as soon as we have enough data
        while sum(len(b) for b in buffer) >= INTERNAL_CHUNK_SAMPLES:
            # Extract exactly 512 samples
            extracted = []
            needed = INTERNAL_CHUNK_SAMPLES
            while needed > 0 and buffer:
                part = buffer.popleft()
                if len(part) <= needed:
                    extracted.append(part)
                    needed -= len(part)
                else:
                    extracted.append(part[:needed])
                    buffer.appendleft(part[needed:])
                    needed = 0
            internal_chunk = np.concatenate(extracted)

            # Keep copy for later reconstruction
            current_segment_buffer.append(internal_chunk.copy())

            # 3. Run VAD
            segment = vad(internal_chunk, return_seconds=True)   # None or single dict

            if segment:                                          # speech segment just finished
                # collect_chunks expects a LIST of dicts → wrap it
                segment_tensor = collect_chunks([segment], torch.from_numpy(internal_chunk))

                if len(current_segment_buffer) > 1:
                    prev = np.concatenate(current_segment_buffer[:-1])
                    full_audio = np.concatenate([prev, segment_tensor.numpy()])
                else:
                    full_audio = segment_tensor.numpy()

                current_segment_buffer.clear()

                if on_segment_complete:
                    on_segment_complete(segment, full_audio)