# base_client_stream.py
import asyncio
import json
import httpx

from typing import List, Optional, AsyncIterator
from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


console = Console()


class TranslationRequest(BaseModel):
    sentences: List[str]


class TranslationResult(BaseModel):
    """Simple container for final translation results."""
    original: str
    translated: str


async def stream_sse_post(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
) -> AsyncIterator[dict]:
    """
    Generic async generator for streaming Server-Sent Events (SSE) from a POST request.
    
    Yields parsed JSON objects from each 'data:' line.
    Stops on '[DONE]' or end of stream.
    
    Args:
        client: Pre-configured httpx.AsyncClient (with desired timeout).
        url: Target endpoint.
        payload: JSON-serializable payload.
    
    Yields:
        dict: Parsed JSON from each valid data line.
    
    Raises:
        RuntimeError: On non-200 response with server error details.
    """
    async with client.stream("POST", url, json=payload) as response:
        if response.status_code != 200:
            text = await response.aread()
            console.print(f"[red]Server error: {response.status_code} - {text.decode()}[/red]")
            raise RuntimeError(f"Request failed: {response.status_code}")

        async for line in response.aiter_lines():
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue

            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break

            try:
                yield json.loads(data_str)
            except json.JSONDecodeError:
                continue


async def collect_full_results(
    sentences: List[str],
    url: str = "http://shawn-pc.local:8001/translate/batch",
    timeout: Optional[float] = None,
) -> List[TranslationResult]:
    """
    Reusable async function to collect only the final completed translations
    from a batch SSE endpoint, without any partial streaming or progress display.

    Designed to mirror the original example_collect_full_results behavior
    but as a clean, generic, reusable utility built on stream_sse_post.

    Args:
        sentences: List of input sentences (used to match order).
        url: SSE endpoint URL.
        timeout: httpx timeout (None recommended for streaming).

    Returns:
        List[TranslationResult]: Completed translations with original text,
        preserving input order.
    """
    payload = TranslationRequest(sentences=sentences).model_dump()
    results: List[str] = [""] * len(sentences)
    current_index: int = -1

    async with httpx.AsyncClient(timeout=timeout or None) as client:
        async for data in stream_sse_post(client=client, url=url, payload=payload):
            # Detect start of new sentence
            if "sentence" in data:
                try:
                    current_index = sentences.index(data["sentence"])
                except ValueError:
                    continue

            # Capture final result
            if "done" in data and current_index >= 0:
                results[current_index] = data["done"]

            # Optional: surface server errors
            if "error" in data:
                console.print(f"[red]Server error for sentence {current_index + 1}: {data['error']}[/red]")

    # Build structured results preserving order
    return [
        TranslationResult(original=orig, translated=trans)
        for orig, trans in zip(sentences, results)
    ]


def translate_ja_en(
    sentences: List[str],
    url: str = "http://shawn-pc.local:8001/translate/batch",
    timeout: Optional[float] = None,
) -> List[str]:
    """
    Synchronous wrapper around the async batch translation endpoint.
    Collects only the final completed translations and returns them in order.

    Args:
        sentences: List of Japanese sentences to translate.
        url: Endpoint URL.
        timeout: httpx timeout (None = no timeout).

    Returns:
        List of English translations in the same order as input.
    """
    return asyncio.run(atranslate_ja_en(sentences=sentences, url=url, timeout=timeout))


async def atranslate_ja_en(
    sentences: List[str],
    url: str = "http://shawn-pc.local:8001/translate/batch",
    timeout: Optional[float] = None,
    *,
    stream_partial: bool = False,
    show_progress: bool = False,
) -> List[str]:
    """
    Reusable async function to translate a batch of Japanese sentences to English.

    Features:
    - Returns only final completed translations in original order.
    - Optional real-time partial token streaming to console.
    - Optional rich/tqdm progress bar per sentence.
    - Generic and reusable – no hard-coded example sentences.

    Args:
        sentences: List of Japanese sentences.
        url: Translation endpoint.
        timeout: httpx timeout (None recommended for streaming).
        stream_partial: If True, print incremental tokens as they arrive.
        show_progress: If True, display a progress bar per sentence.

    Returns:
        List[str]: Completed English translations.
    """
    payload = TranslationRequest(sentences=sentences).model_dump()

    results: List[str] = [""] * len(sentences)
    current_index: int = -1
    pbar: Optional[Progress] = None
    task_id = None  # Will be set when pbar is created

    async with httpx.AsyncClient(timeout=timeout or None) as client:
        async for data in stream_sse_post(client=client, url=url, payload=payload):
            # New sentence detected via original text
            if "sentence" in data:
                try:
                    current_index = sentences.index(data["sentence"])
                except ValueError:
                    continue

            if "partial" in data and stream_partial:
                print(data["partial"], end="", flush=True)

                if show_progress and current_index >= 0 and pbar is None:
                    pbar = Progress(
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        "[progress.percentage]{task.percentage:>3.0f}%",
                        TimeRemainingColumn(),
                    )
                    task_id = pbar.add_task(
                        f"Sentence {current_index + 1}/{len(sentences)}", total=1
                    )
                    pbar.start()

            if "done" in data:
                results[current_index] = data["done"]

                if stream_partial:
                    print()  # Newline after completion

                if show_progress and pbar is not None and task_id is not None:
                    pbar.update(task_id, advance=1)
                    pbar.stop()
                    pbar = None
                    task_id = None

            if "error" in data:
                console.print(f"[red]Server error for sentence {current_index + 1}: {data['error']}[/red]")

    if pbar is not None:
        pbar.stop()

    return results


async def example_sse_streaming_client() -> None:
    """Example: Async streaming client that reuses the generic stream_sse_post function."""
    url = "http://shawn-pc.local:8001/translate/batch"  # Adjust to your FastAPI server URL

    sentences = [
        "世界各国が水面架で知列な情報戦を繰り広げる時代に、にらみ合う2つの国、東のオスタニア、西のウェスタリス、戦",
        "争を加わだてるオスタニア政府要順の動向をさせ、",
    ]

    payload = TranslationRequest(sentences=sentences).model_dump()

    console.print("Streaming translations:\n")

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async for data in stream_sse_post(client=client, url=url, payload=payload):
                if "partial" in data:
                    # Real-time token streaming
                    print(data["partial"], end="", flush=True)
                elif "done" in data:
                    # Final translation for the sentence
                    console.print("\n\n[dim]Original:[/dim]", data["sentence"])
                    print("✓ Completed:", data["done"])
                    console.print("-" * 80)
                elif "error" in data:
                    console.print(f"[red]\nServer error: {data['error']}[/red]")
        except RuntimeError as e:
            console.print(f"[bold red]Request failed:[/bold red] {e}")


async def example_collect_full_results() -> None:
    """
    Updated example using the new reusable collect_full_results function.
    """
    url = "http://shawn-pc.local:8001/translate/batch"

    sentences = [
        "こんにちは、お元気ですか？",
        "今日はとても良い天気ですね。",
        "最近、面白い本を読みました。",
    ]

    completed_results = await collect_full_results(sentences=sentences, url=url)

    console.print("[bold green]All complete translations collected:[/bold green]")
    for result in completed_results:
        console.print(f"[dim]Original:[/dim]  {result.original}")
        console.print(f"[bold]Translated:[/bold] {result.translated}")
        console.print()


async def main() -> None:
    console.print("[bold cyan]=== Example 1: Real-time token streaming ===[/bold cyan]")
    await example_sse_streaming_client()

    console.print("\n[bold cyan]=== Example 2: Collect all complete translations (no streaming) ===[/bold cyan]")
    await example_collect_full_results()


if __name__ == "__main__":
    asyncio.run(main())
