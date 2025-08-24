from typing import List, Sequence
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TaskResult
from autogen_ext.models.replay import ReplayChatCompletionClient
from autogen_core import SingleThreadedAgentRuntime

async def main():
    """
    Demonstrates SelectorGroupChat with streaming output.
    The team processes a task and streams messages as they are generated.
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

    # Initialize SelectorGroupChat with streaming enabled
    team = SelectorGroupChat(
        participants=[agent1, agent2],
        model_client=model_client,
        termination_condition=termination,
        runtime=runtime,
        emit_team_events=True,
        model_client_streaming=True
    )

    # Run the task with streaming
    task = "Write a program that prints 'Hello, world!'"
    print("Streaming Task Results:")
    async for message in team.run_stream(task=task):
        if isinstance(message, TaskResult):
            print(f"Final Result - Stop Reason: {message.stop_reason}")
            for msg in message.messages:
                print(f"{msg.source}: {msg.content}")
        else:
            print(f"{message.source}: {message.content}")

if __name__ == "__main__":
    asyncio.run(main())