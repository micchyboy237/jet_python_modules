"""Client usage examples for the /asr router endpoints using httpx.

These examples demonstrate real-world WebSocket streaming for live Japanese → English
translation using raw 16kHz mono PCM16 audio chunks.

Requirements:
    pip install httpx httpx-ws rich librosa soundfile

Run with: python client_examples/asr_client_examples.py
"""

import asyncio
from pathlib import Path
from typing import AsyncGenerator

import httpx_ws

try:
    from httpx_ws import aconnect_ws  # Modern, maintained WebSocket extension for httpx
except ImportError:  # pragma: no cover
    raise ImportError(
        "WebSocket support requires 'httpx-ws'. Install with: pip install httpx-ws"
    )

import httpx
import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from rich import print as rprint

import numpy as np
import librosa

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
log = logging.getLogger("asr_client")

console = Console()

# WEBSOCKETS_URI = "ws://localhost:8000/asr/live-jp-en"
WEBSOCKETS_URI = "ws://192.168.68.150:8001/asr/live-jp-en"


async def read_audio_chunks(
    audio_path: Path,
    target_sr: int = 16000,
    chunk_duration_sec: float = 5.0,
) -> AsyncGenerator[bytes, None]:
    """
    Load audio with librosa (handles any format, auto-converts to mono and resamples to 16kHz),
    then yield raw int16 PCM16 bytes in fixed-duration chunks.

    This is robust against stereo, different sample rates, or non-WAV formats.
    """
    # Load entire file – efficient for typical short/medium recordings (< few minutes)
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    total_duration = len(y) / target_sr
    total_chunks = int(np.ceil(total_duration / chunk_duration_sec))

    log.info(
        f"[bold cyan]Loaded audio:[/] {audio_path.name} → {total_duration:.2f}s "
        f"({len(y)} samples @ {target_sr}Hz mono) → {total_chunks} chunks of {chunk_duration_sec}s"
    )

    # Convert float32 [-1.0, 1.0] → int16 PCM
    audio_int16 = np.int16(y * 32767.999)  # avoid overflow
    audio_bytes = audio_int16.tobytes()

    bytes_per_chunk = int(target_sr * 2 * chunk_duration_sec)  # 2 bytes per sample

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[bold blue]{task.completed}/{task.total}"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[bold magenta]Streaming audio chunks...", total=total_chunks)

        chunk_idx = 0
        for i in range(0, len(audio_bytes), bytes_per_chunk):
            chunk = audio_bytes[i : i + bytes_per_chunk]
            chunk_samples = len(chunk) // 2  # 2 bytes per int16 sample
            start_sec = chunk_idx * chunk_duration_sec
            end_sec = min(start_sec + chunk_duration_sec, total_duration)
            actual_duration = end_sec - start_sec

            log.debug(
                f"[dim]Chunk {chunk_idx+1:03d}[/] | "
                f"start={start_sec:6.2f}s → end={end_sec:6.2f}s | "
                f"duration={actual_duration:.2f}s | "
                f"samples={chunk_samples:6d} | "
                f"bytes={len(chunk):7d}"
            )

            yield chunk
            progress.update(task, advance=1)
            chunk_idx += 1

    log.info("[bold green]All audio chunks streamed to server[/bold green]")


async def streaming_asr_client(
    audio_path: Path, server_url: str = WEBSOCKETS_URI
):
    """
    Connect to the WebSocket ASR endpoint and stream audio chunks.

    Displays partial and final translations in real-time using rich.
    """
    console.rule("[bold blue]Starting Live Japanese → English ASR Streaming Client[/bold blue]")
    rprint(
        Panel(
            f"Audio file: [cyan]{audio_path.name}[/]\nServer: [magenta]{server_url}[/]"
        )
    )

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with aconnect_ws(server_url, client=client) as websocket:
                log.info("[bold green]WebSocket connected[/bold green]")

                # Start sending audio chunks
                async def send_audio():
                    async for chunk in read_audio_chunks(audio_path):
                        await websocket.send_bytes(chunk)
                    log.info("[yellow]Finished sending audio – closing send channel[/yellow]")
                    await websocket.close()

                # Receive and display results
                async def receive_results():
                    final_segments = []
                    while True:
                        try:
                            message = await websocket.receive_json()
                        except httpx_ws.WebSocketDisconnect:
                            log.info("[bold yellow]Server disconnected cleanly[/bold yellow]")
                            break
                        except httpx_ws.WebSocketNetworkError:
                            log.warning("[bold yellow]Server disconnected due to network error[/bold yellow]")
                            break
                        except Exception as e:
                            log.error(f"[bold red]Unexpected WebSocket error:[/] {e}")
                            break

                        if message.get("final"):
                            text = message["english"]
                            start = message["start"]
                            end = message["end"]
                            duration = end - start
                            final_segments.append((start, end, text))
                            rprint(
                                Panel(
                                    f"[bold green]{text}[/bold green]\n"
                                    f"[dim]Time: {start:6.2f}s → {end:6.2f}s (duration: {duration:.2f}s)[/dim]",
                                    title="FINAL",
                                    border_style="green",
                                )
                            )
                        else:
                            partial = message.get("partial", "")
                            if partial.strip():
                                console.print(f"[dim cyan]Partial:[/] {partial}")

                    # Summary at end
                    if final_segments:
                        console.rule("[bold green]All Final Segments[/bold green]")
                        for start, end, text in final_segments:
                            rprint(f"[green]{start:6.2f}s - {end:6.2f}s[/green]: {text}")

                # Run send and receive concurrently
                await asyncio.gather(send_audio(), receive_results())

    except httpx.ConnectError:
        log.error(f"[bold red]Could not connect to server at {server_url}[/bold red]")
        rprint("[red]Is the FastAPI server running on port 8001?[/red]")
    except Exception as exc:
        log.exception(f"[bold red]Unexpected error:[/] {exc}")


async def main():
    # Replace with your own 16kHz mono 16-bit WAV file path
    example_audio = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/generated/run_extract_speech_timestamps/segments/segment_002/sound.wav")

    if not example_audio.exists():
        rprint(f"[red]Example audio file not found: {example_audio}[/red]")
        rprint(
            "[yellow]Please provide a 16kHz mono 16-bit WAV file and update the path.[/yellow]"
        )
        return

    await streaming_asr_client(example_audio)


if __name__ == "__main__":
    asyncio.run(main())