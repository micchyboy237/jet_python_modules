import asyncio
import os

from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url=os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:1234/v1"),
    api_key="sk-1234",
)


async def main():
    stream = await client.completions.create(
        model=os.getenv("LLAMA_CPP_LLM_MODEL", "not-needed"),
        prompt="What is OpenTelemetry? Answer in exactly one short sentence. Be concise.\nAnswer:",
        max_tokens=128,
        temperature=0.7,
        top_p=0.95,
        presence_penalty=1.5,
        stream_options={"include_usage": True},
        stream=True,
    )

    async for part in stream:
        text = part.choices[0].text or ""
        print(text, end="", flush=True)

        usage = getattr(part, "usage", None)
        if usage is not None:
            print(
                f"\n\nUsage Info:"
                f"\nPrompt tokens: {usage.prompt_tokens}"
                f"\nCompletion tokens: {usage.completion_tokens}"
                f"\nTotal tokens: {usage.total_tokens}"
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
