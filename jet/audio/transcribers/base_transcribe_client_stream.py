#!/usr/bin/env python3
"""
Client examples for the Whisper CTranslate2 + faster-whisper FastAPI server.
All examples are gathered in a single file for convenience.

Run individual examples by calling the corresponding async function from asyncio.run().

Requirements:
    pip install httpx
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx


BASE_URL = "http://127.0.0.1:8001"


async def batch_transcribe_example() -> None:
    """Example 1: Batch transcription of multiple audio files."""
    audio_paths = [
        Path("audio1.wav"),
        Path("audio2.mp3"),
        Path("audio3.ogg"),
        # Add more paths as needed
    ]

    # Field names can be arbitrary – the server accepts any files in the multipart form
    files = {
        f"file{i}": (path.name, open(path, "rb"), "audio/wav")
        for i, path in enumerate(audio_paths)
        if path.exists()
    }

    if not files:
        print("No audio files found.")
        return

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            f"{BASE_URL}/batch/transcribe",
            files=files,
        )

    response.raise_for_status()
    results = response.json()

    print("\nBatch Transcription Results:\n" + "=" * 40)
    for i, res in enumerate(results, 1):
        print(f"File {i}:")
        print(f"  Duration:      {res['duration_sec']}s")
        print(f"  Language:      {res['detected_language']} (prob: {res['detected_language_prob']})")
        print(f"  Transcription: {res['transcription']}\n")


async def batch_transcribe_translate_example() -> None:
    """Example 2: Batch transcription + translation of multiple audio files."""
    audio_paths = [
        Path("spanish1.wav"),
        Path("french2.mp3"),
        # Add more paths as needed
    ]

    files = {
        f"audio{i}": (path.name, open(path, "rb"))
        for i, path in enumerate(audio_paths)
        if path.exists()
    }

    if not files:
        print("No audio files found.")
        return

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            f"{BASE_URL}/batch/transcribe_translate",
            files=files,
        )

    response.raise_for_status()
    results = response.json()

    print("\nBatch Transcribe + Translate Results:\n" + "=" * 50)
    for i, res in enumerate(results, 1):
        print(f"File {i}:")
        print(f"  Transcription: {res['transcription']}")
        print(f"  Translation:   {res.get('translation', 'N/A')}\n")


async def streaming_transcribe_example() -> None:
    """Example 3: Streaming transcription (if /transcribe_stream endpoint is implemented)."""
    audio_path = Path("live_audio.wav")
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return

    async with httpx.AsyncClient(timeout=None) as client:
        with open(audio_path, "rb") as f:
            response = await client.post(
                f"{BASE_URL}/transcribe_stream",  # Adjust endpoint if different
                content=f,
                headers={"Content-Type": "audio/wav"},
            )

    response.raise_for_status()

    print("\nStreaming Transcription:\n")
    async for line in response.aiter_lines():
        if line.strip():
            try:
                data = httpx.json.loads(line)
                text = data.get("text", "").strip()
                if text:
                    print(f"{text}", end=" ", flush=True)
            except Exception:
                print(line, end=" ")
    print("\n\nStreaming complete.")


async def concurrent_batch_requests_example() -> None:
    """Example 4: Running multiple independent batch requests concurrently."""
    batch1 = [Path("audio1.wav"), Path("audio2.wav")]
    batch2 = [Path("audio3.mp3"), Path("audio4.mp3")]

    async def send_batch(files_paths: list[Path], endpoint: str) -> list[dict]:
        files = {
            f"file{i}": (p.name, open(p, "rb"))
            for i, p in enumerate(files_paths)
            if p.exists()
        }
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(f"{BASE_URL}/batch/{endpoint}", files=files)
            resp.raise_for_status()
            return resp.json()

    tasks = [
        send_batch(batch1, "transcribe"),
        send_batch(batch2, "transcribe_translate"),
    ]

    results = await asyncio.gather(*tasks)

    print("\nConcurrent Batch Results:\n" + "=" * 40)
    print("Batch 1 (transcribe):")
    for res in results[0]:
        print(f"  → {res['transcription'][:80]}...")
    print("\nBatch 2 (transcribe+translate):")
    for res in results[1]:
        print(f"  → {res['transcription'][:60]}... → {res.get('translation', 'N/A')[:60]}...")


if __name__ == "__main__":
    # Uncomment the example you want to run
    # asyncio.run(batch_transcribe_example())
    # asyncio.run(batch_transcribe_translate_example())
    # asyncio.run(streaming_transcribe_example())
    # asyncio.run(concurrent_batch_requests_example())
    print("Uncomment the desired example in __main__ to run it.")