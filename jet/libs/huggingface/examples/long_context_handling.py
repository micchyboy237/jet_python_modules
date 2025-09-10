"""Manage long context in an AssistantAgent with AutoGen v0.4.

This module demonstrates how to handle long message histories in AutoGen v0.4 using `BufferedChatCompletionContext` with an `AssistantAgent`. It replaces the v0.2 `transforms` capability, limiting the model’s view to the last 10 messages to manage context window constraints in a chatbot scenario.
"""

import asyncio
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_ext.models.ollama import OllamaChatCompletionClient


async def main() -> None:
    model_client = OllamaChatCompletionClient(
        model="llama3.2", seed=42, temperature=0)
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        model_client=model_client,
        model_context=BufferedChatCompletionContext(buffer_size=10),
    )
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        response = await assistant.on_messages([TextMessage(content=user_input, source="user")], CancellationToken())
        print("Assistant:", response.chat_message.to_text())
    await model_client.close()

asyncio.run(main())
