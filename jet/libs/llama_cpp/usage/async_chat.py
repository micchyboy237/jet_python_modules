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
    response = await client.chat.completions.create(
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
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
