import asyncio
import os
import shutil
import yaml
import warnings
from typing import Sequence
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.utils import content_to_str
from autogen_core.models import ModelFamily
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.base import TerminationCondition, TerminatedException
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import TextMessage, BaseAgentEvent, BaseChatMessage, HandoffMessage, MultiModalMessage, StopMessage
from autogen_core.models import LLMMessage, UserMessage, AssistantMessage

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

FILE_BASE_PATH = "/Users/jethroestrada/Desktop/External_Projects/AI"

LOGS_DIR = f"{OUTPUT_DIR}/logs"
DOWNLOADS_DIR = f"{OUTPUT_DIR}/downloads"
CODING_DIR = f"{OUTPUT_DIR}/coding"

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(CODING_DIR, exist_ok=True)

# Suppress warnings about the requests.Session() not being closed
warnings.filterwarnings(
    action="ignore", message="unclosed", category=ResourceWarning)


async def main() -> None:
    model_client = OllamaChatCompletionClient(model="llama3.2")
    orchestrator_client = model_client
    coder_client = model_client
    web_surfer_client = model_client
    file_surfer_client = model_client

    # Updated prompt to search for files related to RAG agents
    prompt = """Navigate the file system starting from the base path to find files related to RAG (Retrieval-Augmented Generation) agents for long documents. Focus on Python files (*.py) or configuration files (*.yaml, *.yml) that contain terms like 'RAG', 'retrieval', 'augmented', or 'long document'. Use available tools (e.g., open_path, find_on_page_ctrl_f) to search directories and file contents. Return a comma-separated list of file names (without paths) that match these criteria."""

    # Set up the team
    coder = MagenticOneCoderAgent(
        "Assistant",
        model_client=coder_client,
    )

    executor = CodeExecutorAgent(
        "ComputerTerminal",
        code_executor=LocalCommandLineCodeExecutor(
            work_dir=CODING_DIR,
            cleanup_temp_files=False
        )
    )

    file_surfer = FileSurfer(
        name="FileSurfer",
        model_client=file_surfer_client,
        base_path=FILE_BASE_PATH
    )

    web_surfer = MultimodalWebSurfer(
        name="WebSurfer",
        model_client=web_surfer_client,
        headless=False,
        downloads_folder=DOWNLOADS_DIR,
        debug_dir=LOGS_DIR,
        to_save_screenshots=True,
    )

    # Prepare the prompt
    task = prompt.strip()

    # Updated termination conditions
    max_messages_termination = MaxMessageTermination(max_messages=20)
    llm_termination = LLMTermination(
        prompt=f"""Consider the following task:
{task}

Does the above conversation suggest that a comma-separated list of file names containing RAG agent implementations or configurations for long documents has been provided?
If so, reply "TERMINATE", otherwise reply "CONTINUE"
""",
        model_client=orchestrator_client
    )

    termination = max_messages_termination | llm_termination

    # Create the team
    team = SelectorGroupChat(
        [coder, executor, file_surfer, web_surfer],
        model_client=orchestrator_client,
        termination_condition=termination,
    )

    # Run the task
    stream = team.run_stream(task=task)
    result = await Console(stream)

    # Do one more inference to format the results
    final_context: Sequence[LLMMessage] = []
    for message in result.messages:
        if isinstance(message, TextMessage):
            final_context.append(UserMessage(
                content=message.content, source=message.source))
        elif isinstance(message, MultiModalMessage):
            if orchestrator_client.model_info["vision"]:
                final_context.append(UserMessage(
                    content=message.content, source=message.source))
            else:
                final_context.append(UserMessage(content=content_to_str(
                    message.content), source=message.source))
    final_context.append(UserMessage(
        content=f"""We have completed the following task:
{prompt}

The above messages contain the conversation that took place to complete the task.
Read the above conversation and output a FINAL ANSWER to the question.
To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a comma-separated list of file names (without paths) that contain RAG agent implementations or configurations for long documents.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., comma-separated list, no articles, no punctuation).
#""".strip(),
        source="user"))

    # Call the model to evaluate
    response = await orchestrator_client.create(final_context)
    print(response.content, flush=True)


class LLMTermination(TerminationCondition):
    """Terminate the conversation if an LLM determines the task is complete.

    Args:
        prompt: The prompt to evaluate in the LLM to check if the task is complete.
        model_client: The LLM model client to use for evaluation.
        termination_phrase: The phrase to look for in the LLM output to trigger termination (default: "TERMINATE").
    """

    def __init__(self, prompt: str, model_client: OllamaChatCompletionClient, termination_phrase: str = "TERMINATE") -> None:
        self._prompt = prompt
        self._model_client = model_client
        self._termination_phrase = termination_phrase
        self._terminated = False
        self._context: Sequence[LLMMessage] = []

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> StopMessage | None:
        if self._terminated:
            raise TerminatedException(
                "Termination condition has already been reached")

        # Build the context
        for message in messages:
            if isinstance(message, TextMessage):
                self._context.append(UserMessage(
                    content=message.content, source=message.source))
            elif isinstance(message, MultiModalMessage):
                if self._model_client.model_info["vision"]:
                    self._context.append(UserMessage(
                        content=message.content, source=message.source))
                else:
                    self._context.append(UserMessage(content=content_to_str(
                        message.content), source=message.source))

        if len(self._context) == 0:
            return None

        # Call the model to evaluate
        response = await self._model_client.create(self._context + [UserMessage(content=self._prompt, source="user")])

        # Check for termination
        if isinstance(response.content, str) and self._termination_phrase in response.content:
            self._terminated = True
            return StopMessage(content=response.content, source="LLMTermination")
        return None

    async def reset(self) -> None:
        self._terminated = False
        self._context = []


if __name__ == "__main__":
    asyncio.run(main())
