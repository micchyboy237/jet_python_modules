from typing import List, Sequence
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_ext.models.replay import ReplayChatCompletionClient
from autogen_core import SingleThreadedAgentRuntime

async def main():
    """
    Demonstrates a basic SelectorGroupChat with two agents and a termination condition.
    The team selects agents to respond to a task until a termination condition is met.
    """
    # Initialize runtime
    runtime = SingleThreadedAgentRuntime()

    # Mock model client for reproducible results
    model_client = ReplayChatCompletionClient(["agent2", "agent1", "agent2", "agent1"])

    # Create agents
    agent1 = AssistantAgent(
        name="agent1",
        model_client=model_client,
        description="Assistant for general tasks",
        system_message="You are a helpful assistant."
    )
    agent2 = AssistantAgent(
        name="agent2",
        model_client=model_client,
        description="Assistant for technical queries",
        system_message="You provide technical assistance."
    )

    # Define termination condition
    termination = TextMentionTermination("TERMINATE")

    # Initialize SelectorGroupChat
    team = SelectorGroupChat(
        participants=[agent1, agent2],
        model_client=model_client,
        termination_condition=termination,
        runtime=runtime,
        allow_repeated_speaker=True
    )

    # Run the task
    task = "Write a program that prints 'Hello, world!'"
    result = await team.run(task=task)

    # Print results
    print("Task Result:")
    for message in result.messages:
        print(f"{message.source}: {message.content}")
    print(f"Stop Reason: {result.stop_reason}")

if __name__ == "__main__":
    asyncio.run(main())