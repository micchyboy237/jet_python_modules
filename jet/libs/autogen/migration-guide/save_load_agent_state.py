"""Save and load an AssistantAgent's state in AutoGen v0.4.

This module demonstrates how to save and load the state of an `AssistantAgent` in AutoGen v0.4, replacing the v0.2 manual handling of `chat_messages`. It shows how to persist the agent's chat history to a JSON file and restore it to continue or revert conversations, with asynchronous message handling.
"""

import asyncio
import json
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient


async def main() -> None:
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o", seed=42, temperature=0)
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        model_client=model_client,
    )
    cancellation_token = CancellationToken()
    response = await assistant.on_messages([TextMessage(content="Hello!", source="user")], cancellation_token)
    print(response)
    state = await assistant.save_state()
    with open("assistant_state.json", "w") as f:
        json.dump(state, f)
    with open("assistant_state.json", "r") as f:
        state = json.load(f)
        print(state)
    response = await assistant.on_messages([TextMessage(content="Tell me a joke.", source="user")], cancellation_token)
    print(response)
    await assistant.load_state(state)
    response = await assistant.on_messages([TextMessage(content="Tell me a joke.", source="user")], cancellation_token)
    await model_client.close()

asyncio.run(main())
