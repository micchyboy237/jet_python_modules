from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, AsyncGenerator  # <-- updated import

import httpx
from rich import print as rprint

BASE_URL = "http://shawn-pc.local:8001/transcribe_translate"

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

        # Updated: Skip raise_for_status on redirects (3xx) or informational (1xx)
        async def raise_on_4xx_5xx(response: httpx.Response):
            # Only raise on client/server errors (4xx/5xx); allow 1xx/2xx/3xx
            if response.is_client_error or response.is_server_error:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    rprint(f"[bold red]HTTP error: {exc}[/bold red]")
                    raise
            # Optional: Log non-errors for debugging
            else:
                rprint(f"[dim cyan]Response: {response.status_code} {response.reason_phrase}[/dim cyan]")

        _client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=timeout,
            limits=limits,
            http2=True,
            headers={"User-Agent": "audio-transcriber/1.0"},
            follow_redirects=True,  # Explicitly enable
            event_hooks={
                "response": [raise_on_4xx_5xx]
            }
        )
    return _client

async def upload_file_multipart(file_path: Path) -> Dict[str, Any]:
    client = await get_client()
    # Read synchronously once – perfectly fine for <100 MB files
    data = file_path.read_bytes()
    files = {
        "data": (
            file_path.name,
            data,  # ← bytes directly
            "application/octet-stream",
        )
    }
    rprint(f"[bold green]Sending as multipart:[/bold green] {file_path} ({len(data):,} bytes)")
    r = await client.post("", files=files)
    # New: Explicitly raise on final response (post-redirects) for 4xx/5xx
    r.raise_for_status()
    return r.json()

# --- OPTIONAL: truly async streaming version (for >500 MB files) ---
async def _async_file_chunks(file_path: Path, chunk_size: int = 65536) -> AsyncGenerator[bytes, None]:
    """Yield file chunks asynchronously – works with any size."""
    def _iterator():
        with file_path.open("rb") as f:
            while chunk := f.read(chunk_size):
                yield chunk
    for chunk in _iterator():   # runs synchronously but yields fast enough for network
        yield chunk


async def upload_file_multipart_streaming(file_path: Path) -> Dict[str, Any]:
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
    # New: Explicitly raise on final response
    r.raise_for_status()
    return r.json()

async def upload_raw_bytes(data: bytes) -> Dict[str, Any]:
    client = await get_client()
    rprint(f"[bold yellow]Sending as raw bytes:[/bold yellow] {len(data):,} bytes")
    r = await client.post(
        "",
        content=data,
        headers={"Content-Type": "application/octet-stream"},
    )
    # New: Explicitly raise on final response
    r.raise_for_status()
    return r.json()

async def close_client() -> None:
    """Call this on app shutdown or script exit."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None

async def main() -> None:
    file_path = Path(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/"
        "python_scripts/samples/audio/data/sound.wav"
    )

    start = time.perf_counter()
    result1 = await upload_file_multipart(file_path)
    mid = time.perf_counter()
    rprint(json.dumps(result1, indent=2, ensure_ascii=False))
    rprint(f"[bold green]multipart duration:[/bold green] {mid - start:.3f}s")

    rprint("\n" + "─" * 50 + "\n")

    start2 = time.perf_counter()
    data = file_path.read_bytes()
    result2 = await upload_raw_bytes(data)
    end = time.perf_counter()
    rprint(json.dumps(result2, indent=2, ensure_ascii=False))
    rprint(f"[bold yellow]raw bytes duration:[/bold yellow] {end - start2:.3f}s")

    # Total time (useful when batching many files)
    rprint(f"[bold cyan]Total elapsed:[/bold cyan] {end - start:.3f}s")

    # Always close cleanly
    await close_client()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Guarantees cleanup even on Ctrl+C
        asyncio.run(close_client())