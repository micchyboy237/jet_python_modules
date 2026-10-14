import sys

sys.path.insert(
    0,
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/microsoft-agent-framework/python/packages/core",
)
sys.path.insert(
    0,
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/microsoft-agent-framework/python/packages/openai",
)
import asyncio

from agent_framework import Agent, Message
from agent_framework.openai import OpenAIChatCompletionClient
from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL


async def main():
    client = OpenAIChatCompletionClient(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key="not-needed",
    )
    agent = Agent(
        client=client,
        name="LocalLlamaAgent",
        instructions="You are a helpful assistant.",
    )

    # Create a list of messages
    messages = [
        Message(role="user", contents=[{"type": "text", "text": "Hello!"}]),
        Message(
            role="assistant",
            contents=[{"type": "text", "text": "Hi there! How can I help?"}],
        ),
        Message(role="user", contents=[{"type": "text", "text": "What is AI?"}]),
    ]

    response = await agent.run(messages)
    print(response.text)


if __name__ == "__main__":
    asyncio.run(main())
