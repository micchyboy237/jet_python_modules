import os
import shutil
import asyncio
from typing import Sequence
import uuid
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.ui import Console

from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Define agent tools for content creation


async def write_content(topic: str) -> str:
    return f"Draft content for {topic}: Engaging article with key points."


async def edit_content(content: str) -> str:
    return f"Edited content: {content} with improved clarity and tone."


async def main() -> None:
    model_client = OllamaChatCompletionClient(
        model="llama3.2", host="http://localhost:11434")

    # Create agents with specific roles
    writer = AssistantAgent(
        name="Writer",
        model_client=model_client,
        tools=[write_content],
        description="Creates initial content drafts.",
    )
    editor = AssistantAgent(
        name="Editor",
        model_client=model_client,
        tools=[edit_content],
        description="Edits and refines content drafts.",
    )

    # Define a custom selector function
    def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        if not messages:
            return "Writer"  # Start with Writer
        last_message = messages[-1]
        if isinstance(last_message, BaseChatMessage):
            if last_message.source == "Writer":
                return "Editor"  # After Writer, select Editor
            if last_message.source == "Editor":
                return "Writer"  # After Editor, select Writer
        return None

    # Define termination condition
    termination = TextMentionTermination("FINALIZED")

    # Create the SelectorGroupChat team with custom selector function
    team = SelectorGroupChat(
        participants=[writer, editor],
        model_client=model_client,
        selector_func=selector_func,
        termination_condition=termination,
        max_turns=3,
        selector_prompt="""You are managing a content creation team. The following roles are available:
{roles}.
Select the next role from {participants} to respond. Only return the role name.
{history}""",
        allow_repeated_speaker=False,
    )

    # Run the team with a content creation task
    task = "Create a blog post about sustainable energy solutions."
    await Console(team.run_stream(task=task))

    state = await team.save_state()
    save_file(state, f"{OUTPUT_DIR}/team_state.json")

if __name__ == "__main__":
    asyncio.run(main())
