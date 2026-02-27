from pathlib import Path

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
    context_sec: float = 1.0,  # recommended retained context for next buffer payload
    min_silence_for_completion_sec: float = 0.35,
    **extract_kwargs,
) -> tuple[list, np.ndarray, dict]:
    """
    Modular helper to:
    - Optionally append new chunk to utterance_audio_buffer
    - Extract all speech segments via extract_speech_timestamps
    - Return: all completed speech segments, updated buffer trimmed accordingly, payload dict
    - Retain context after submitted speech (for VAD stability in next call)
    """
    # Step 1: build current buffer (optionally concatenate)
    if new_audio_chunk is not None and len(new_audio_chunk) > 0:
        chunk_np = np.asarray(new_audio_chunk, dtype=np.float32)
        if len(utterance_audio_buffer) == 0:
            current_audio = chunk_np
        else:
            current_audio = np.concatenate([utterance_audio_buffer, chunk_np])
    else:
        current_audio = np.asarray(utterance_audio_buffer, dtype=np.float32).copy()

    if len(current_audio) == 0:
        payload = {
            "speech_segments": [],
            "total_buffer_duration": 0.0,
            "submitted_count": 0,
        }
        return [], current_audio, payload

    total_duration = len(current_audio) / sampling_rate

    # Step 2: reuse existing extractor (DRY) + debug
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

    # Step 3: collect completed speech + detect ongoing (for correct trimming)
    console.print(
        f"[bold blue][DEBUG] VAD returned {len(all_segments)} segments[/bold blue]"
    )
    submitted_segments = []
    last_submitted_end = 0.0
    has_ongoing_speech = False
    for seg in all_segments:
        if seg.get("type") == "speech":
            end_sec = float(seg["end"])
            completion_check = (
                end_sec + min_silence_for_completion_sec <= total_duration
            )
            console.print(
                f"[DEBUG] Speech seg {seg['num']}: {end_sec:.2f}s → completion_check={completion_check}"
            )
            if completion_check:
                submitted_segments.append(dict(seg))  # copy for safety
                last_submitted_end = end_sec
            else:
                has_ongoing_speech = True
                console.print(
                    "[yellow][DEBUG] Ongoing speech detected at buffer end[/yellow]"
                )
                break  # stop at ongoing speech

    payload = {
        "speech_segments": submitted_segments,  # <-- sends all completed segments
        "total_buffer_duration": round(total_duration, 3),
        "submitted_count": len(submitted_segments),
    }

    # Step 4: trim buffer (pure idle vs ongoing vs completed)
    console.print(
        f"[bold blue][DEBUG] submitted={len(submitted_segments)} | has_ongoing={has_ongoing_speech}[/bold blue]"
    )
    if submitted_segments:
        retain_start_sample = int(last_submitted_end * sampling_rate)
        updated_buffer = current_audio[retain_start_sample:].copy()
        console.print(
            f"[green][DEBUG] Trimmed after last submitted end={last_submitted_end:.2f}s[/green]"
        )
    elif has_ongoing_speech:
        updated_buffer = current_audio.copy()
        console.print("[yellow][DEBUG] Ongoing speech → keep FULL buffer[/yellow]")
    else:
        # pure idle
        max_samples = int(context_sec * sampling_rate)
        updated_buffer = (
            current_audio[-max_samples:]
            if len(current_audio) > max_samples
            else current_audio.copy()
        )
        console.print(f"[DEBUG] Pure idle → capped to context_sec={context_sec}s")

    console.print(
        f"[bold green]Payload ready[/bold green] | Submitted: {len(submitted_segments)} speech segments | New buffer: {len(updated_buffer)} samples (~{len(updated_buffer) / sampling_rate:.2f}s)"
    )

    return submitted_segments, updated_buffer, payload


class StreamingSpeechProcessor:
    """Reusable high-level processor for real-time VAD streaming.
    Maintains the utterance_audio_buffer internally and exposes one clean
    .process(chunk) call site. Perfect for microphone loops, WebSocket handlers,
    or local servers.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        context_sec: float = 1.0,  # Recommended retained context for next utterance_audio_buffer payload
        min_silence_for_completion_sec: float = 0.35,
        **extract_kwargs,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.context_sec = context_sec
        self.min_silence_for_completion_sec = min_silence_for_completion_sec
        self.extract_kwargs = extract_kwargs
        self.utterance_audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        console.print(
            "[bold cyan]✅ StreamingSpeechProcessor ready (context_sec=1.0s)[/bold cyan]"
        )

    def process(self, new_audio_chunk: np.ndarray | None = None) -> dict:
        """Call this on every incoming audio chunk.
        Returns payload with ALL completed speech segments (replace your old payload here).
        Buffer is automatically trimmed (submitted speech removed).
        """
        submitted, self.utterance_audio_buffer, payload = process_utterance_buffer(
            utterance_audio_buffer=self.utterance_audio_buffer,
            new_audio_chunk=new_audio_chunk,
            sampling_rate=self.sampling_rate,
            context_sec=self.context_sec,
            min_silence_for_completion_sec=self.min_silence_for_completion_sec,
            **self.extract_kwargs,
        )
        return payload

    def reset(self) -> None:
        """Clear buffer (e.g. after session end or silence timeout)."""
        self.utterance_audio_buffer = np.array([], dtype=np.float32)
        console.print("[yellow]Buffer reset[/yellow]")


if __name__ == "__main__":
    # Original file-processing demo (unchanged)
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"
    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_file).name}")
    segments = extract_speech_timestamps(
        audio_file,
        threshold=0.5,
        neg_threshold=0.25,
        max_speech_duration_sec=8.0,
        return_seconds=True,
        time_resolution=2,
        normalize_loudness=False,
    )
    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments)}\n")
    for seg in segments:
        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white] - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"duration=[bold magenta]{seg['duration']}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan]"
        )

    # NEW: Streaming usage demo (copy this pattern into your real loop)
    console.rule("[bold cyan]Streaming integration example[/bold cyan]")
    processor = StreamingSpeechProcessor(
        threshold=0.5,
        neg_threshold=0.25,
        max_speech_duration_sec=8.0,
        context_sec=1.0,
    )
    # In your code replace the loop below with real mic/websocket chunks:
    # for chunk in audio_stream_generator():
    #     payload = processor.process(chunk)
    #     if payload["submitted_count"] > 0:
    #         your_send_function(payload)   # <-- sends ALL speech segments
    console.print(
        "[bold green]✅ Copy the .process() call above into your streaming loop[/bold green]"
    )
