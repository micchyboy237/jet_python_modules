# client_examples.py
import asyncio
import json
from typing import List

import httpx
from pydantic import BaseModel
from rich.console import Console

from jet.audio.transcribers.base_translate_client_stream import atranslate_ja_en


console = Console()


class TranslationRequest(BaseModel):
    sentences: List[str]


async def sse_streaming_client(sentences: List[str]) -> None:
    """Real-time token streaming using the reusable async client."""

    console.print("Streaming translations:\n")

    # We manually parse the stream to show incremental tokens and final results
    url = "http://shawn-pc.local:8001/translate/batch"

    payload = TranslationRequest(sentences=sentences)

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload.model_dump()) as response:
            if response.status_code != 200:
                text = await response.aread()
                console.print(f"[red]Server error: {response.status_code} - {text.decode()}[/red]")
                return

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    if "partial" in data:
                        print(data["partial"], end="", flush=True)
                    elif "done" in data:
                        print("\n\nOriginal:", data["sentence"])
                        print("Translated:", data["done"])
                        print("-" * 80)
                    elif "error" in data:
                        print("\nServer error:", data["error"])
                except json.JSONDecodeError:
                    continue


async def collect_full_results(sentences: List[str]) -> None:
    """
    Collect all complete translations into a list using the reusable function.
    """

    # Simple and clean – just await the reusable function
    results: List[str] = await atranslate_ja_en(sentences)

    console.print("[bold]All complete translations collected:[/bold]")
    for original, translated in zip(sentences, results):
        console.print(f"[bold]Original:[/bold]  {original}")
        console.print(f"[bold]Translated:[/bold] {translated}\n")


async def with_progress_bar(sentences: List[str]) -> None:
    """Streaming with per-sentence progress bar using reusable client."""

    console.print("[bold]Translating with progress per sentence...[/bold]\n")

    # Use the reusable function with streaming and progress enabled
    await atranslate_ja_en(
        sentences=sentences,
        stream_partial=True,
        show_progress=True,
    )


async def main() -> None:
    sentences = [
        "世界各国が水面下で熾烈な情報戦を繰り広げる時代に、にらみ合う2つの国、東のオスタニア、西のウェスタリス、",
        "戦争を回避するため、オスタニア政府要人の動向を探るスパイが暗躍していた。",
        "その中でも特に優秀なスパイ、黄昏と呼ばれる男がいた。",
    ]

    console.print("[bold cyan]=== Real-time token streaming ===[/bold cyan]")
    await sse_streaming_client(sentences)

    sentences = [
        "こんにちは、お元気ですか？",
        "今日はとても良い天気ですね。",
        "最近、面白い本を読みました。",
    ]
    console.print("\n[bold cyan]=== Example 2: Collect all results ===[/bold cyan]")
    await collect_full_results(sentences)

    sentences = [
        "スパイファミリーは面白いアニメです。",
        "ロイドは優秀なスパイです。",
        "アーニャは可愛いです。",
        "ヨルは強いです。",
    ]
    console.print("\n[bold cyan]=== Example 3: With progress bar per sentence ===[/bold cyan]")
    await with_progress_bar(sentences)


if __name__ == "__main__":
    asyncio.run(main())