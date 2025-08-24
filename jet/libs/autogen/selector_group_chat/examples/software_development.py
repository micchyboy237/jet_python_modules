import asyncio
import os
import shutil
from typing import List, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from jet.file.utils import save_file
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter
from jet.logger import logger
from jet.transformers.formatters import format_json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


async def main() -> None:
    # model_client = MLXAutogenChatLLMAdapter(
    #     model="llama-3.2-3b-instruct-4bit", log_dir=f"{OUTPUT_DIR}/chats")
    model_client = OllamaChatCompletionClient(
        model="llama3.2", host="http://localhost:11434")

    filtered_participants = ["developer", "tester"]

    def dummy_candidate_func(thread: Sequence[BaseAgentEvent | BaseChatMessage]) -> List[str]:
        # Dummy candidate function that will return
        # only return developer and reviewer
        return filtered_participants

    developer = AssistantAgent(
        "developer",
        description="Writes and implements code based on requirements.",
        model_client=model_client,
        system_message="You are a software developer working on a new feature.",
    )

    tester = AssistantAgent(
        "tester",
        description="Writes and executes test cases to validate the implementation.",
        model_client=model_client,
        system_message="You are a software tester ensuring the feature works correctly.",
    )

    project_manager = AssistantAgent(
        "project_manager",
        description="Oversees the project and ensures alignment with the broader goals.",
        model_client=model_client,
        system_message="You are a project manager ensuring the team meets the project goals.",
    )

    team = SelectorGroupChat(
        participants=[developer, tester, project_manager],
        model_client=model_client,
        max_turns=3,
        candidate_func=dummy_candidate_func,
    )

    task = "Create a detailed implementation plan for adding dark mode in a React app and review it for feasibility and improvements."

    messages = []
    async for message in team.run_stream(task=task):
        logger.debug(f"Message:\n{format_json(message)}")
        messages.append(message)
        save_file(messages, f"{OUTPUT_DIR}/messages.json")

        if not isinstance(message, TaskResult):
            if message.source == "user":  # ignore the first 'user' message
                continue
            assert message.source in filtered_participants, "Candidate function didn't filter the participants"

    state = await team.save_state()
    save_file(state, f"{OUTPUT_DIR}/team_state.json")

if __name__ == "__main__":
    asyncio.run(main())
