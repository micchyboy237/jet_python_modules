"""Create and use an AssistantAgent in AutoGen v0.4 for message handling.

This module demonstrates how to instantiate an `AssistantAgent` in AutoGen v0.4, replacing the v0.2 `AssistantAgent`. It configures the agent with a model client and system message, and processes messages asynchronously using `on_messages`, supporting cancellation via `CancellationToken`.
"""

import asyncio
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o", seed=42, temperature=0)
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        model_client=model_client,
    )
    cancellation_token = CancellationToken()
    response = await assistant.on_messages([TextMessage(content="Hello!", source="user")], cancellation_token)
    print(response)
    await model_client.close()

asyncio.run(main())