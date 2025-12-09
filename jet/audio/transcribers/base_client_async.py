from __future__ import annotations

import asyncio
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, AsyncGenerator, Union, TypedDict

import httpx
from rich import print as rprint

BASE_URL = "http://shawn-pc.local:8001/transcribe_translate"

AudioInput = Union[str, Path, np.ndarray]

# ---- Typed Dict for Response ----
class TranscribeResponse(TypedDict):
    duration_sec: float
    detected_language: str
    detected_language_prob: float
    transcription: str
    translation: str

# Global client – created once and reused for the entire process lifetime
_client: Optional[httpx.AsyncClient] = None

async def get_client() -> httpx.AsyncClient:
    """Lazy-initialize a high-performance AsyncClient with pooling & HTTP/2."""
    global _client
    if _client is None:
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0,
        )
        timeout = httpx.Timeout(30.0, connect=10.0)

        async def raise_on_4xx_5xx(response: httpx.Response):
            if response.is_client_error or response.is_server_error:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    rprint(f"[bold red]HTTP error: {exc}[/bold red]")
                    raise
            else:
                rprint(f"[dim cyan]Response: {response.status_code} {response.reason_phrase}[/dim cyan]")

        _client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=timeout,
            limits=limits,
            http2=True,
            headers={"User-Agent": "audio-transcriber/1.0"},
            follow_redirects=True,
            event_hooks={
                "response": [raise_on_4xx_5xx]
            }
        )
    return _client

async def upload_file_multipart(file_path: Path) -> TranscribeResponse:
    client = await get_client()
    data = file_path.read_bytes()
    files = {
        "data": (
            file_path.name,
            data,
            "application/octet-stream",
        )
    }
    rprint(f"[bold green]Sending as multipart:[/bold green] {file_path} ({len(data):,} bytes)")
    r = await client.post("", files=files)
    r.raise_for_status()
    result = r.json()
    return result  # runtime type; conforms to TranscribeResponse if server as expected

async def _async_file_chunks(file_path: Path, chunk_size: int = 65536) -> AsyncGenerator[bytes, None]:
    def _iterator():
        with file_path.open("rb") as f:
            while chunk := f.read(chunk_size):
                yield chunk
    for chunk in _iterator():
        yield chunk

async def upload_file_multipart_streaming(file_path: Union[str, Path]) -> TranscribeResponse:
    file_path = Path(file_path)
    client = await get_client()

    async def _streaming_generator():
        async for chunk in _async_file_chunks(file_path):
            yield chunk

    files = {
        "data": (
            file_path.name,
            _streaming_generator(),
            "application/octet-stream",
        )
    }
    rprint(f"[bold green]Streaming multipart:[/bold green] {file_path}")
    r = await client.post("", files=files)
    r.raise_for_status()
    result = r.json()
    return result  # runtime type; conforms to TranscribeResponse if server as expected

async def upload_raw_bytes(data: bytes) -> TranscribeResponse:
    client = await get_client()
    rprint(f"[bold yellow]Sending as raw bytes:[/bold yellow] {len(data):,} bytes")
    r = await client.post(
        "",
        content=data,
        headers={"Content-Type": "application/octet-stream"},
    )
    r.raise_for_status()
    result = r.json()
    return result  # runtime type; conforms to TranscribeResponse if server as expected

async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None

async def atranscribe_audio(audio: AudioInput) -> TranscribeResponse:
    """Async entry-point – works with file paths, in-memory bytes or NumPy arrays."""
    if isinstance(audio, (str, Path)):
        file_path = Path(audio)
        return await upload_file_multipart(file_path)

    if isinstance(audio, (bytes, bytearray)):
        return await upload_raw_bytes(audio)

    if isinstance(audio, np.ndarray):
        data = audio.tobytes()
        return await upload_raw_bytes(data)

    raise TypeError(
        f"Unsupported audio input type: {type(audio)!r}. "
        "Expected Path/str, bytes, bytearray, or np.ndarray."
    )

async def main() -> None:
    file_path = Path(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/"
        "python_scripts/samples/audio/data/sound.wav"
    )

    start = time.perf_counter()
    result1 = await atranscribe_audio(file_path)
    mid = time.perf_counter()
    rprint(json.dumps(result1, indent=2, ensure_ascii=False))
    rprint(f"[bold green]multipart duration:[/bold green] {mid - start:.3f}s")

    rprint("\n" + "─" * 50 + "\n")

    start2 = time.perf_counter()
    data = file_path.read_bytes()
    result2 = await atranscribe_audio(data)
    end = time.perf_counter()
    rprint(json.dumps(result2, indent=2, ensure_ascii=False))
    rprint(f"[bold yellow]raw bytes duration:[/bold yellow] {end - start2:.3f}s")

    rprint(f"[bold cyan]Total elapsed:[/bold cyan] {end - start:.3f}s")

    await close_client()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(close_client())