from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, Union, Optional

import httpx
import numpy as np
from rich import print as rprint
from mimetypes import guess_type

from jet.audio.transcribers.base import AudioInput

import asyncio

BASE_URL = "http://shawn-pc.local:8001/transcribe_translate"
# BASE_URL = "http://shawn-pc.local:8001/transcribe_translate_kotoba"


class TranscribeResponse(TypedDict):
    duration_sec: float
    detected_language: str
    detected_language_prob: float
    transcription: str
    translation: str


# Global asynchronous client – created once and reused
_async_client: Optional[httpx.AsyncClient] = None


def _guess_mime(path_or_filename: Union[str, Path]) -> str:
    mime, _ = guess_type(str(path_or_filename))
    return mime or "application/octet-stream"


async def get_async_client() -> httpx.AsyncClient:
    """Lazy-initialize a high-performance AsyncClient with pooling & HTTP/2."""
    global _async_client
    if _async_client is None:
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0,
        )
        timeout = httpx.Timeout(
            timeout=300.0,
            connect=15.0,
            read=300.0,
            write=300.0,
            pool=30.0,
        )

        async def raise_on_4xx_5xx(response: httpx.Response):
            if response.is_client_error or response.is_server_error:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    rprint(f"[bold red]HTTP error: {exc}[/bold red]")
                    raise
            else:
                rprint(f"[dim cyan]Response: {response.status_code} {response.reason_phrase}[/dim cyan]")

        _async_client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=timeout,
            limits=limits,
            http2=True,
            headers={"User-Agent": "audio-transcriber/1.0"},
            follow_redirects=True,
            event_hooks={"response": [raise_on_4xx_5xx]},
        )
    return _async_client


async def atranscribe_audio(
    audio: AudioInput,
    *,
    filename: str = "sound.wav"
) -> TranscribeResponse:
    """
    Asynchronous entry-point for transcription/translation.

    Accepts:
        - File path (str / Path)
        - Raw bytes / bytearray
        - NumPy array (converted with .tobytes())
        - Filename for in-memory data (for correct MIME type)

    Returns parsed TranscribeResponse dict.
    """
    client = await get_async_client()

    # ------------------------------------------------------------------
    # 1. File path → multipart upload (best for large files on disk)
    # ------------------------------------------------------------------
    if isinstance(audio, (str, Path)):
        file_path = Path(audio)
        if not file_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        mime_type = _guess_mime(file_path)

        rprint(f"[bold green]Sending multipart ({mime_type}):[/bold green] {file_path}")
        # FIX: Use normal file open context, as httpx.AsyncStream does not exist.
        # Safe to use blocking file open here because file uploads are rare and
        # httpx.AsyncClient will upload in background without blocking event loop.
        with file_path.open("rb") as f:
            files = {
                "data": (file_path.name, f, mime_type)
            }
            response = await client.post("", files=files)

    # ------------------------------------------------------------------
    # 2. In-memory bytes / bytearray / np.ndarray
    # ------------------------------------------------------------------
    else:
        if isinstance(audio, np.ndarray):
            data = audio.tobytes()
        elif isinstance(audio, (bytes, bytearray)):
            data = bytes(audio)
        else:
            raise TypeError(
                f"Unsupported audio input type: {type(audio)!r}. "
                "Expected Path/str, bytes, bytearray, or np.ndarray."
            )

        if filename:
            mime_type = _guess_mime(filename)
            rprint(f"[bold cyan]Sending in-memory as multipart ({mime_type}):[/bold cyan] {filename} ({len(data):,} bytes)")
            files = {
                "data": (filename, data, mime_type)
            }
            response = await client.post("", files=files)
        else:
            mime_type = "application/octet-stream"
            rprint(f"[bold yellow]Sending raw bytes ({mime_type}):[/bold yellow] {len(data):,} bytes")
            response = await client.post(
                "",
                content=data,
                headers={"Content-Type": mime_type},
            )

    # ------------------------------------------------------------------
    # Common response handling
    # ------------------------------------------------------------------
    response.raise_for_status()
    result: TranscribeResponse = response.json()
    return result


async def aclose_async_client() -> None:
    """Close the global asynchronous client and free resources."""
    global _async_client
    if _async_client is not None:
        await _async_client.aclose()
        _async_client = None


# ----------------------------------------------------------------------
# Example usage (async main)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import time

    async def main():
        try:
            file_path = Path(
                "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_20251222_125319.wav"
            )

            # Test with file path
            start1 = time.perf_counter()
            result1 = await atranscribe_audio(file_path)
            end1 = time.perf_counter()
            rprint("[bold green]Multipart result:[/bold green]")
            rprint(json.dumps(result1, indent=2, ensure_ascii=False))
            print(f"[bold green]upload_file_multipart duration:[/bold green] {end1 - start1:.3f} seconds")

            # Test with in-memory bytes
            raw_data = file_path.read_bytes()
            start2 = time.perf_counter()
            result2 = await atranscribe_audio(raw_data)
            end2 = time.perf_counter()
            rprint("[bold yellow]Raw bytes result:[/bold yellow]")
            rprint(json.dumps(result2, indent=2, ensure_ascii=False))
            print(f"[bold yellow]upload_raw_bytes duration:[/bold yellow] {end2 - start2:.3f} seconds")

        finally:
            await aclose_async_client()

    asyncio.run(main())