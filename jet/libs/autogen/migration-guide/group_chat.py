"""Set up a group chat with writer and critic agents in AutoGen v0.4.

This module demonstrates a group chat in AutoGen v0.4 using `RoundRobinGroupChat` with `AssistantAgent` instances for a writer and critic. It replaces the v0.2 `GroupChatManager`, alternating between agents to collaboratively write and critique a story, with termination based on a specific message.
"""

import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o", seed=42, temperature=0)
    writer = AssistantAgent(
        name="writer",
        description="A writer.",
        system_message="You are a writer.",
        model_client=model_client,
    )
    critic = AssistantAgent(
        name="critic",
        description="A critic.",
        system_message="You are a critic, provide feedback on the writing. Reply only 'APPROVE' if the task is done.",
        model_client=model_client,
    )
    termination = TextMentionTermination("APPROVE")
    group_chat = RoundRobinGroupChat([writer, critic], termination_condition=termination, max_turns=12)
    stream = group_chat.run_stream(task="Write a short story about a robot that discovers it has feelings.")
    await Console(stream)
    await model_client.close()

asyncio.run(main())