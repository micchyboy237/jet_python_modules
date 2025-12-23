"""Client usage examples for the /asr router endpoints using httpx.

These examples demonstrate real-world WebSocket streaming for live Japanese → English
translation using raw 16kHz mono PCM16 audio chunks.

Requirements:
    pip install httpx httpx-ws rich wave

Run with: python client_examples/asr_client_examples.py
"""

import asyncio
import wave
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
from rich import print as rprint

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
    audio_path: Path, chunk_duration_sec: float = 5.0
) -> AsyncGenerator[bytes, None]:
    """
    Read a WAV file and yield raw PCM16 chunks matching the expected server format.

    Assumes input is mono 16kHz 16-bit PCM (standard for faster-whisper streaming).
    """
    with wave.open(str(audio_path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError("Audio must be mono")
        if wf.getframerate() != 16000:
            raise ValueError("Audio must be 16kHz")
        if wf.getsampwidth() != 2:
            raise ValueError("Audio must be 16-bit PCM")

        bytes_per_chunk = int(16000 * 2 * chunk_duration_sec)  # 2 bytes per sample

        while True:
            chunk = wf.readframes(bytes_per_chunk // 2)
            if not chunk:
                break
            yield chunk


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
                        log.debug(f"[dim]Sent chunk: {len(chunk)} bytes[/dim]")
                    log.info("[yellow]Finished sending audio – closing send channel[/yellow]")
                    await websocket.close()

                # Receive and display results
                async def receive_results():
                    final_segments = []
                    while True:
                        try:
                            message = await websocket.receive_json()
                        except httpx_ws.WebSocketDisconnect:
                            log.info("[bold yellow]Server disconnected[/bold yellow]")
                            break

                        if message.get("final"):
                            text = message["english"]
                            start = message["start"]
                            end = message["end"]
                            final_segments.append((start, end, text))
                            rprint(
                                Panel(
                                    f"[bold green]{text}[/bold green]\n"
                                    f"[dim]Time: {start:.2f}s → {end:.2f}s[/dim]",
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
    example_audio = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav")

    if not example_audio.exists():
        rprint(f"[red]Example audio file not found: {example_audio}[/red]")
        rprint(
            "[yellow]Please provide a 16kHz mono 16-bit WAV file and update the path.[/yellow]"
        )
        return

    await streaming_asr_client(example_audio)


if __name__ == "__main__":
    asyncio.run(main())