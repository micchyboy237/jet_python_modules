from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, Union, Optional

import httpx
import numpy as np
from rich import print as rprint

BASE_URL = "http://shawn-pc.local:8001/transcribe_translate"

AudioInput = Union[str, Path, bytes, bytearray, np.ndarray]

# ---- Typed Dict for Response ----
class TranscribeResponse(TypedDict):
    duration_sec: float
    detected_language: str
    detected_language_prob: float
    transcription:         str
    translation:           str


# Global synchronous client – created once and reused
_client: Optional[httpx.Client] = None


def get_sync_client() -> httpx.Client:
    """Lazy-initialize a high-performance synchronous Client with pooling & HTTP/2."""
    global _client
    if _client is None:
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0,
        )
        timeout = httpx.Timeout(30.0, connect=10.0)

        def raise_on_4xx_5xx(response: httpx.Response):
            if response.is_client_error or response.is_server_error:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    rprint(f"[bold red]HTTP error: {exc}[/bold red]")
                    raise
            else:
                rprint(f"[dim cyan]Response: {response.status_code} {response.reason_phrase}[/dim cyan]")

        _client = httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            limits=limits,
            http2=True,
            headers={"User-Agent": "audio-transcriber/1.0"},
            follow_redirects=True,
            event_hooks={"response": [raise_on_4xx_5xx]},
        )
    return _client


def transcribe_audio(audio: AudioInput) -> TranscribeResponse:
    """
    Synchronous entry-point for transcription.

    Accepts:
        - File path (str / Path)
        - Raw bytes / bytearray
        - NumPy array (will be converted with .tobytes())

    Returns parsed TranscribeResponse dict.
    """
    client = get_sync_client()

    # ------------------------------------------------------------------
    # 1. File path → multipart upload (best for large files on disk)
    # ------------------------------------------------------------------
    if isinstance(audio, (str, Path)):
        file_path = Path(audio)
        if not file_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        rprint(f"[bold green]Sending multipart:[/bold green] {file_path} ({file_path.stat().st_size:,} bytes)")
        with file_path.open("rb") as f:
            files = {
                "data": (
                    file_path.name,
                    f,
                    "application/octet-stream",
                )
            }
            response = client.post("", files=files)

    # ------------------------------------------------------------------
    # 2. In-memory bytes / bytearray / np.ndarray → raw binary body
    # ------------------------------------------------------------------
    else:
        if isinstance(audio, np.ndarray):
            data: bytes = audio.tobytes()
        elif isinstance(audio, (bytes, bytearray)):
            data = bytes(audio)
        else:
            raise TypeError(
                f"Unsupported audio input type: {type(audio)!r}. "
                "Expected Path/str, bytes, bytearray, or np.ndarray."
            )

        rprint(f"[bold yellow]Sending raw bytes:[/bold yellow] {len(data):,} bytes")
        response = client.post(
            "",
            content=data,
            headers={"Content-Type": "application/octet-stream"},
        )

    # ------------------------------------------------------------------
    # Common response handling
    # ------------------------------------------------------------------
    response.raise_for_status()
    result: TranscribeResponse = response.json()
    return result


def close_sync_client() -> None:
    """Close the global synchronous client and free resources."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


# ----------------------------------------------------------------------
# Example usage (can be placed in if __name__ == "__main__": block)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import time
    try:
        file_path = Path(
            "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/"
            "python_scripts/samples/audio/data/sound.wav"
        )
        # Test with file path
        start1 = time.perf_counter()
        result1 = transcribe_audio(file_path)
        end1 = time.perf_counter()
        rprint("[bold green]Multipart result:[/bold green]")
        rprint(json.dumps(result1, indent=2, ensure_ascii=False))
        print(f"[bold green]upload_file_multipart duration:[/bold green] {end1 - start1:.3f} seconds")

        # Test with in-memory bytes
        raw_data = file_path.read_bytes()
        start2 = time.perf_counter()
        result2 = transcribe_audio(raw_data)
        end2 = time.perf_counter()
        rprint("[bold yellow]Raw bytes result:[/bold yellow]")
        rprint(json.dumps(result2, indent=2, ensure_ascii=False))
        print(f"[bold yellow]upload_raw_bytes duration:[/bold yellow] {end2 - start2:.3f} seconds")

    finally:
        close_sync_client()