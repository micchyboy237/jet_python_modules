# client_examples.py
import asyncio
import json
from typing import List

import httpx
from pydantic import BaseModel


class TranslationRequest(BaseModel):
    sentences: List[str]


async def example_sse_streaming_client() -> None:
    """Example: Async streaming client using EventSource-like parsing with httpx."""
    url = "http://shawn-pc.local:8001/translate/batch"  # Adjust to your FastAPI server URL

    payload = TranslationRequest(
        sentences=[
            "世界各国が水面下で熾烈な情報戦を繰り広げる時代に、にらみ合う2つの国、東のオスタニア、西のウェスタリス、",
            "戦争を回避するため、オスタニア政府要人の動向を探るスパイが暗躍していた。",
            "その中でも特に優秀なスパイ、黄昏と呼ばれる男がいた。",
        ]
    )

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload.model_dump()) as response:
            if response.status_code != 200:
                text = await response.aread()
                print(f"Server error: {response.status_code} - {text.decode()}")
                return

            print("Streaming translations:\n")
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "partial" in data:
                            # Show incremental tokens (no newline to simulate real-time typing)
                            print(data["partial"], end="", flush=True)
                        elif "done" in data:
                            # Final translation for the sentence
                            print("\n\n✓ Completed:", data["done"])
                            print("Original:", data["sentence"])
                            print("-" * 80)
                        elif "error" in data:
                            print("\nServer error:", data["error"])
                    except json.JSONDecodeError:
                        continue


async def example_collect_full_results() -> None:
    """
    Example: Collect all complete translations into a list.
    Useful when you want full results before further processing.
    """
    url = "http://shawn-pc.local:8001/translate/batch"

    payload = TranslationRequest(
        sentences=[
            "こんにちは、お元気ですか？",
            "今日はとても良い天気ですね。",
            "最近、面白い本を読みました。",
        ]
    )

    results: List[str] = []

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload.model_dump()) as response:
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                    if "done" in data:
                        results.append(data["done"])
                except json.JSONDecodeError:
                    continue

    print("All complete translations collected:")
    for original, translated in zip(payload.sentences, results):
        print(f"Original:  {original}")
        print(f"Translated: {translated}\n")


async def example_with_progress_bar() -> None:
    """Example: Using tqdm to show progress per sentence."""
    from tqdm.asyncio import tqdm_asyncio

    url = "http://shawn-pc.local:8001/translate/batch"

    sentences = [
        "スパイファミリーは面白いアニメです。",
        "ロイドは優秀なスパイです。",
        "アーニャは可愛いです。",
        "ヨルは強いです。",
    ]

    payload = TranslationRequest(sentences=sentences)

    tasks = []

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload.model_dump()) as response:
            sentence_index = -1
            pbar = None

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                    if "partial" in data and "sentence" in data:
                        # Detect new sentence start
                        current_index = sentences.index(data["sentence"])
                        if current_index != sentence_index:
                            if pbar:
                                pbar.close()
                            sentence_index = current_index
                            pbar = tqdm_asyncio(
                                total=1,
                                desc=f"Translating [{sentence_index+1}/{len(sentences)}]",
                                unit="sentence",
                                leave=False,
                            )
                        print(data["partial"], end="", flush=True)
                    elif "done" in data:
                        if pbar:
                            pbar.update(1)
                            pbar.close()
                        print("\n")
                except Exception:
                    continue


async def main() -> None:
    print("=== Example 1: Real-time token streaming ===")
    await example_sse_streaming_client()

    print("\n=== Example 2: Collect all results ===")
    await example_collect_full_results()

    # Uncomment to run progress bar example
    print("\n=== Example 3: With tqdm progress per sentence ===")
    await example_with_progress_bar()


if __name__ == "__main__":
    asyncio.run(main())