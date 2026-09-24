import asyncio
import os

from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url=os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:1234/v1"),
    api_key="sk-1234",
)


async def main():
    messages = [
        {
            "role": "user",
            "content": "Write a 2 sentence short story about a curious robot.",
        },
    ]
    stream = await client.chat.completions.create(
        model="qwen3-instruct-2507:4b",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        presence_penalty=1.5,
        stream_options={"include_usage": True},
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
        stream=True,
    )

    async for part in stream:
        if part.choices and part.choices[0].delta:
            delta = part.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                print(delta.reasoning_content, end="", flush=True)
            elif hasattr(delta, "content") and delta.content:
                print(delta.content, end="", flush=True)

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
