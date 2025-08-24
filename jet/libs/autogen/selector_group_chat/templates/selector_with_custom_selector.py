from typing import List, Sequence
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_ext.models.replay import ReplayChatCompletionClient
from autogen_core import SingleThreadedAgentRuntime

async def main():
    """
    Demonstrates SelectorGroupChat with a custom selector function to control agent selection logic.
    The custom selector chooses the next agent based on the previous message source.
    """
    # Initialize runtime
    runtime = SingleThreadedAgentRuntime()

    # Mock model client
    model_client = ReplayChatCompletionClient(["agent3"])

    # Create agents
    agent1 = AssistantAgent(
        name="agent1",
        model_client=model_client,
        description="First assistant",
        system_message="You respond to initial tasks."
    )
    agent2 = AssistantAgent(
        name="agent2",
        model_client=model_client,
        description="Second assistant",
        system_message="You follow up on agent1's responses."
    )
    agent3 = AssistantAgent(
        name="agent3",
        model_client=model_client,
        description="Third assistant",
        system_message="You handle specific tasks."
    )
    agent4 = AssistantAgent(
        name="agent4",
        model_client=model_client,
        description="Fourth assistant",
        system_message="You finalize responses."
    )

    # Custom selector function
    def custom_selector(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        if not messages:
            return "agent1"
        last_source = messages[-1].source
        if last_source == "agent1":
            return "agent2"
        elif last_source == "agent2":
            return None
        elif last_source == "agent3":
            return "agent4"
        return "agent1"

    # Define termination condition
    termination = MaxMessageTermination(max_messages=6)

    # Initialize SelectorGroupChat with custom selector
    team = SelectorGroupChat(
        participants=[agent1, agent2, agent3, agent4],
        model_client=model_client,
        selector_func=custom_selector,
        termination_condition=termination,
        runtime=runtime
    )

    # Run the task
    task = "Plan a team meeting agenda."
    result = await team.run(task=task)

    # Print results
    print("Task Result:")
    for message in result.messages:
        print(f"{message.source}: {message.content}")
    print(f"Stop Reason: {result.stop_reason}")

if __name__ == "__main__":
    asyncio.run(main())