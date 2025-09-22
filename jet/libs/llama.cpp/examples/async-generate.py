import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")  # Dummy API key
    response = await client.completions.create(
        model="ggml-org/gemma-3-4b-it-GGUF",
        prompt="Why is the sky blue?",
    )
    print(response.choices[0].text)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")