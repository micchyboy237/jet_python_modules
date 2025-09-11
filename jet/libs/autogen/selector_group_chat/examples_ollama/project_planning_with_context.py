import os
import shutil
import asyncio
import uuid
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient
from autogen_agentchat.ui import Console

from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Define agent tools for project planning


async def assign_task(task: str, assignee: str) -> str:
    return f"Task '{task}' assigned to {assignee}."


async def estimate_time(task: str) -> str:
    return f"Estimated time for '{task}' is 3 days."


async def review_plan(plan: str) -> str:
    return f"Plan '{plan}' looks feasible, approved."


async def main() -> None:
    # Initialize a buffered model context to keep only the last 3 messages
    model_context = BufferedChatCompletionContext(buffer_size=3)

    model_client = OllamaChatCompletionClient(
        model="llama3.2", host="http://localhost:11434")

    # Create agents with specific roles
    project_manager = AssistantAgent(
        name="Project_Manager",
        model_client=model_client,
        description="Coordinates project tasks and assigns responsibilities.",
        tools=[assign_task],
    )
    analyst = AssistantAgent(
        name="Analyst",
        model_client=model_client,
        description="Estimates task durations and resource needs.",
        tools=[estimate_time],
    )
    reviewer = AssistantAgent(
        name="Reviewer",
        model_client=model_client,
        description="Reviews and approves project plans.",
        tools=[review_plan],
    )

    # Define termination condition
    termination = TextMentionTermination("APPROVED")

    # Create the SelectorGroupChat team with custom model context
    team = SelectorGroupChat(
        participants=[project_manager, analyst, reviewer],
        model_client=model_client,
        termination_condition=termination,
        max_turns=4,
        model_context=model_context,
        selector_prompt="""You are managing a project planning team. The following roles are available:
{roles}.
Based on the conversation, select the next role from {participants} to respond. Only return the role name.
{history}""",
        allow_repeated_speaker=True,
    )

    # Run the team with a project planning task
    task = "Plan a new feature development for our app, including task assignment and time estimation."
    await Console(team.run_stream(task=task))

    state = await team.save_state()
    save_file(state, f"{OUTPUT_DIR}/team_state.json")

if __name__ == "__main__":
    asyncio.run(main())
