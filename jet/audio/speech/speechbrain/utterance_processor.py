# jet_python_modules/jet/audio/speech/speechbrain/utterance_processor.py
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from jet.audio.speech.speechbrain.speech_timestamps_extractor import (
    extract_speech_timestamps,
)
from jet.audio.speech.utils import display_segments
from rich.console import Console

console = Console()


def process_utterance_buffer(
    utterance_audio_buffer: np.ndarray,
    new_audio_chunk: np.ndarray | None = None,
    sampling_rate: int = 16000,
    context_sec: float = 1.0,
    min_silence_for_completion_sec: float = 0.35,
    **extract_kwargs,
) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    """
    Process utterance buffer and return:
    - list of completed (submittable) speech segments
    - updated (trimmed) utterance buffer for next iteration
    - payload dictionary for downstream consumers
    - optional snapshot of the buffer *before trimming* (only when segments were submitted)
    """
    if new_audio_chunk is not None and len(new_audio_chunk) > 0:
        chunk_np = np.asarray(new_audio_chunk, dtype=np.float32)
        if len(utterance_audio_buffer) == 0:
            current_audio = chunk_np
        else:
            current_audio = np.concatenate([utterance_audio_buffer, chunk_np])
    else:
        current_audio = np.asarray(utterance_audio_buffer, dtype=np.float32).copy()

    if len(current_audio) == 0:
        payload: Dict[str, Any] = {
            "speech_segments": [],
            "total_buffer_duration": 0.0,
            "submitted_count": 0,
        }
        return [], current_audio, payload, None

    total_duration = len(current_audio) / sampling_rate
    console.print(
        f"[bold blue][DEBUG] total_duration={total_duration:.2f}s | samples={len(current_audio)}[/bold blue]"
    )

    all_segments = extract_speech_timestamps(
        current_audio,
        sampling_rate=sampling_rate,
        include_non_speech=False,
        return_seconds=True,
        **extract_kwargs,
    )
    display_segments(all_segments)
    console.print(
        f"[bold blue][DEBUG] VAD returned {len(all_segments)} segments[/bold blue]"
    )

    submitted_segments: List[Dict[str, Any]] = []
    last_submitted_end = 0.0
    has_ongoing_speech = False

    for seg in all_segments:
        if seg.get("type") == "speech":
            end_sec = float(seg["end"])
            completion_check = (
                end_sec + min_silence_for_completion_sec <= total_duration
            )
            console.print(
                f"[DEBUG] Speech seg {seg.get('num', '?')}: {end_sec:.2f}s → completion_check={completion_check}"
            )
            if completion_check:
                submitted_segments.append(dict(seg))
                last_submitted_end = end_sec
            else:
                has_ongoing_speech = True
                console.print(
                    "[yellow][DEBUG] Ongoing speech detected at buffer end[/yellow]"
                )
                break

    payload = {
        "speech_segments": submitted_segments,
        "total_buffer_duration": round(total_duration, 3),
        "submitted_count": len(submitted_segments),
    }

    console.print(
        f"[bold blue][DEBUG] submitted={len(submitted_segments)} | has_ongoing={has_ongoing_speech}[/bold blue]"
    )

    pre_trim_snapshot: Optional[np.ndarray] = None

    if submitted_segments:
        # Snapshot *before* we trim — used by caller to save correct audio regions
        pre_trim_snapshot = current_audio.copy()

        retain_start_sample = int(last_submitted_end * sampling_rate)
        updated_buffer = current_audio[retain_start_sample:].copy()
        console.print(
            f"[green][DEBUG] Trimmed after last submitted end={last_submitted_end:.2f}s[/green]"
        )
    elif has_ongoing_speech:
        updated_buffer = current_audio.copy()
        console.print("[yellow][DEBUG] Ongoing speech → keep FULL buffer[/yellow]")
    else:
        # Pure silence / idle — keep small context window
        max_samples = int(context_sec * sampling_rate)
        updated_buffer = (
            current_audio[-max_samples:]
            if len(current_audio) > max_samples
            else current_audio.copy()
        )
        console.print(f"[DEBUG] Pure idle → capped to context_sec={context_sec}s")

    console.print(
        f"[bold green]Payload ready[/bold green] | "
        f"Submitted: {len(submitted_segments)} speech segments | "
        f"New buffer: {len(updated_buffer)} samples (~{len(updated_buffer) / sampling_rate:.2f}s)"
    )

    return submitted_segments, updated_buffer, payload, pre_trim_snapshot


class StreamingSpeechProcessor:
    """High-level processor for real-time streaming VAD.

    Maintains internal audio buffer and exposes clean .process() interface.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        context_sec: float = 1.0,
        min_silence_for_completion_sec: float = 0.35,
        **extract_kwargs,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.context_sec = context_sec
        self.min_silence_for_completion_sec = min_silence_for_completion_sec
        self.extract_kwargs = extract_kwargs

        self.utterance_audio_buffer: np.ndarray = np.array([], dtype=np.float32)

        console.print(
            f"[bold cyan]✅ StreamingSpeechProcessor ready "
            f"(sr={sampling_rate}, context={context_sec}s)[/bold cyan]"
        )

    def process(
        self, new_audio_chunk: np.ndarray | None = None
    ) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """
        Process one audio chunk.

        Returns:
            (payload: dict, pre_trim_buffer: np.ndarray | None)
            pre_trim_buffer is only non-None when one or more segments were completed this call
        """
        _, self.utterance_audio_buffer, payload, pre_trim_buffer = (
            process_utterance_buffer(
                utterance_audio_buffer=self.utterance_audio_buffer,
                new_audio_chunk=new_audio_chunk,
                sampling_rate=self.sampling_rate,
                context_sec=self.context_sec,
                min_silence_for_completion_sec=self.min_silence_for_completion_sec,
                **self.extract_kwargs,
            )
        )

        return payload, pre_trim_buffer

    def reset(self) -> None:
        """Clear internal buffer (e.g. after long silence or session end)."""
        self.utterance_audio_buffer = np.array([], dtype=np.float32)
        console.print("[yellow]Buffer reset[/yellow]")
