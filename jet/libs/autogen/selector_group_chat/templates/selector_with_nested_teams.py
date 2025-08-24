from typing import List, Sequence
import asyncio
import tempfile
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_ext.models.replay import ReplayChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_core import SingleThreadedAgentRuntime

async def main():
    """
    Demonstrates SelectorGroupChat with a nested RoundRobinGroupChat team.
    The outer team selects between an inner team (assistant + code executor) and a reviewer agent.
    """
    # Initialize runtime
    runtime = SingleThreadedAgentRuntime()

    # Mock model client for reproducible results
    model_client = ReplayChatCompletionClient([
        "InnerTeam",
        '```python\nprint("Hello, world!")\n```',
        "TERMINATE",
        "agent3",
        "Good job",
        "TERMINATE"
    ])

    # Create temporary directory for code execution
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create code executor
        code_executor = LocalCommandLineCodeExecutor(work_dir=temp_dir)

        # Create agents for inner team
        assistant = AssistantAgent(
            name="assistant",
            model_client=model_client,
            description="Writes code",
            system_message="You write code based on requirements."
        )
        code_executor_agent = CodeExecutorAgent(
            name="code_executor",
            code_executor=code_executor
        )

        # Create inner team
        inner_team = RoundRobinGroupChat(
            participants=[assistant, code_executor_agent],
            termination_condition=TextMentionTermination("TERMINATE"),
            runtime=runtime,
            name="InnerTeam",
            description="Team that writes and executes code"
        )

        # Create reviewer agent
        reviewer = AssistantAgent(
            name="agent3",
            model_client=model_client,
            description="Reviews code",
            system_message="You review code for correctness."
        )

        # Create outer team
        outer_team = SelectorGroupChat(
            participants=[inner_team, reviewer],
            model_client=model_client,
            termination_condition=TextMentionTermination("TERMINATE"),
            runtime=runtime
        )

        # Run the task
        task = "Write a program that prints 'Hello, world!'"
        result = await outer_team.run(task=task)

        # Print results
        print("Task Result:")
        for message in result.messages:
            print(f"{message.source}: {message.content}")
        print(f"Stop Reason: {result.stop_reason}")

if __name__ == "__main__":
    asyncio.run(main())