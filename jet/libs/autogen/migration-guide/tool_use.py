"""Create a tool-using AssistantAgent in AutoGen v0.4 for interactive chat.

This module demonstrates how to configure an `AssistantAgent` in AutoGen v0.4 to handle both tool calling and execution, replacing the v0.2 two-agent approach with a single agent. It supports asynchronous tool usage (e.g., weather lookup) and interactive user input with reflection on tool results.
"""

import asyncio
from autogen_core import CancellationToken
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter


def get_weather(city: str) -> str:
    return f"The weather in {city} is 72 degree and sunny."


async def main() -> None:
    model_client = MLXAutogenChatLLMAdapter(
        model="llama-3.2-3b-instruct-4bit", seed=42, temperature=0)
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant. You can call tools to help user.",
        model_client=model_client,
        tools=[get_weather],
        reflect_on_tool_use=True,
    )
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        response = await assistant.on_messages([TextMessage(content=user_input, source="user")], CancellationToken())
        print("Assistant:", response.chat_message.to_text())
    await model_client.close()

asyncio.run(main())
