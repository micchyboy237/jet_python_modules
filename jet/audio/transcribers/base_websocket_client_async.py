from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, Dict, Any, Callable, Optional

import websockets
from rich.console import Console
from rich.table import Table

console = Console()

WS_URL = "ws://shawn-pc.local:8001/ws/live_transcribe_translate"


async def stream_audio_to_websocket(
    audio_chunks: AsyncGenerator[bytes, None],
    *,
    client_id: str,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_format: str = "int16",  # or "float32"
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """
    Streams audio chunks to the WebSocket server and yields/results incrementally.
    """
    async with websockets.connect(WS_URL) as ws:
        # Send initial configuration
        await ws.send(json.dumps({
            "sample_rate": sample_rate,
            "channels": channels,
            "format": sample_format,
            "client_id": client_id,
        }))
        console.print(f"[bold green]WebSocket connected – streaming started (client_id={client_id})[/bold green]")

        # Start listening for server messages
        async def receive_loop():
            try:
                async for message in ws:
                    if isinstance(message, str):
                        data: Dict[str, Any] = json.loads(message)
                    else:
                        data = {"error": "Received binary message unexpectedly"}

                    # Optional: filter by client_id (safety check)
                    if data.get("client_id") != client_id:
                        console.print(f"[dim]Ignoring message for different client: {data.get('client_id')}[/dim]")
                        continue

                    if on_result:
                        on_result(data)
                    else:
                        _default_print(data)
            except websockets.ConnectionClosed:
                console.print("[yellow]Server closed connection[/yellow]")

        receive_task = asyncio.create_task(receive_loop())

        # Send audio chunks
        try:
            async for chunk in audio_chunks:
                if len(chunk) == 0:
                    continue
                await ws.send(chunk)  # binary frame
        finally:
            await ws.close()
            receive_task.cancel()


def _default_print(result: Dict[str, Any]) -> None:
    result_type = result.get("type", "unknown")
    prefix = "[bold blue]PARTIAL[/]" if result_type == "partial" else "[bold green]FINAL[/]"

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Japanese", style="cyan")
    table.add_column("English", style="green")

    ja = result.get("transcription", "").strip()
    en = result.get("translation", "").strip()

    table.add_row(ja or "[dim](empty)[/]", en or "[dim](empty)[/]")
    console.print(f"\n{prefix} Live Translation")
    console.print(table)
    console.print()