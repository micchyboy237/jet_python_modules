import asyncio
from rich.console import Console

from jet.translators.translate_jp_en_gemma import collect_full_results, sse_streaming_client, with_progress_bar

console = Console()

async def main() -> None:
    sentences = [
        "なんだその理屈は",
    ]

    console.print("[bold cyan]=== Real-time token streaming ===[/bold cyan]")
    await sse_streaming_client(sentences)

    # sentences = [
    #     "こんにちは、お元気ですか？",
    #     "今日はとても良い天気ですね。",
    #     "最近、面白い本を読みました。",
    # ]
    # console.print("\n[bold cyan]=== Example 2: Collect all results ===[/bold cyan]")
    # await collect_full_results(sentences)

    # sentences = [
    #     "スパイファミリーは面白いアニメです。",
    #     "ロイドは優秀なスパイです。",
    #     "アーニャは可愛いです。",
    #     "ヨルは強いです。",
    # ]
    # console.print("\n[bold cyan]=== Example 3: With progress bar per sentence ===[/bold cyan]")
    # await with_progress_bar(sentences)


if __name__ == "__main__":
    asyncio.run(main())