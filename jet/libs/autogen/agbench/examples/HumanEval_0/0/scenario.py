import asyncio
import os
import shutil
import yaml
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily
from autogen_core.model_context import UnboundedChatCompletionContext, ChatCompletionContext
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_agentchat.conditions import TextMentionTermination
from custom_code_executor import CustomCodeExecutorAgent
from reasoning_model_context import ReasoningModelContext
from autogen_core.models import ChatCompletionClient

from jet.logger import logger, CustomLogger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

WORK_DIR = f"{OUTPUT_DIR}/coding"


async def log_stream_chunks(stream):
    """
    Iterate over the stream and log each chunk with immediate flushing.

    Args:
        stream: Async iterable yielding stream chunks from agent_team.run_stream.
    """
    stream_logger = CustomLogger(
        name="StreamResponseLogger", log_file=os.path.join(OUTPUT_DIR, "stream.log"))
    async for chunk in stream:
        # Convert chunk to string for logging; adjust based on chunk type if needed
        chunk_content = str(chunk)
        stream_logger.teal(chunk_content, flush=True)


async def main() -> None:
    with open(f"{os.path.dirname(__file__)}/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_client = ChatCompletionClient.load_component(config["model_config"])
    model_context: ChatCompletionContext
    if model_client.model_info["family"] == ModelFamily.R1:
        model_context = ReasoningModelContext()
    else:
        model_context = UnboundedChatCompletionContext()

    coder_agent = MagenticOneCoderAgent(
        name="coder",
        model_client=model_client,
        model_client_stream=True,
    )
    coder_agent._model_context = model_context

    os.makedirs(WORK_DIR, exist_ok=True)
    executor = CustomCodeExecutorAgent(
        name="executor",
        code_executor=LocalCommandLineCodeExecutor(work_dir=WORK_DIR),
        sources=["coder"],
    )

    termination = TextMentionTermination(
        text="TERMINATE", sources=["executor"])
    agent_team = RoundRobinGroupChat(
        [coder_agent, executor],
        max_turns=4,
        termination_condition=termination,
    )

    prompt = ""
    with open(f"{os.path.dirname(__file__)}/prompt.txt", "rt") as fh:
        prompt = fh.read()

    task = f"""Complete the following python function. Format your output as Markdown python code block containing the entire function definition:
```python
{prompt}
```
"""
    stream = agent_team.run_stream(task=task)

    # Log stream chunks and display in console
    await log_stream_chunks(stream)
    # Optionally keep Console for display; comment out if only logging is needed
    # await Console(stream)

asyncio.run(main())
