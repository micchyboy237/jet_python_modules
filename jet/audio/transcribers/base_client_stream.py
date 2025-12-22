# translation_client.py
import asyncio
import json
from typing import List, Optional

import httpx
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
    payload = TranslationRequest(sentences=sentences)

    results: List[str] = [""] * len(sentences)  # Pre-allocate to preserve order
    current_index: int = -1
    pbar: Optional[Progress] = None

    async with httpx.AsyncClient(timeout=timeout or None) as client:
        async with client.stream("POST", url, json=payload.model_dump()) as response:
            if response.status_code != 200:
                text = await response.aread()
                console.print(f"[red]Server error: {response.status_code} - {text.decode()}[/red]")
                raise RuntimeError(f"Translation failed: {response.status_code}")

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # New sentence detected via original text
                if "sentence" in data:
                    try:
                        current_index = sentences.index(data["sentence"])
                    except ValueError:
                        continue

                if "partial" in data and stream_partial:
                    # Rich Console.print does not support end="" or flush=True → use built-in print for real-time streaming
                    print(data["partial"], end="", flush=True)

                    if show_progress and current_index >= 0:
                        if pbar is None:
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
                        print()  # Newline after completion using built-in print for consistency

                    if show_progress and pbar is not None:
                        pbar.update(task_id, advance=1)
                        pbar.stop()
                        pbar = None

                if "error" in data:
                    console.print(f"[red]Server error for sentence {current_index + 1}: {data['error']}[/red]")

    if pbar is not None:
        pbar.stop()

    return results