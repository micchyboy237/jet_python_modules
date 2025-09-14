import os
from pathlib import Path
import shutil
import asyncio
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient
from jet.logger import logger
from jet.transformers.formatters import format_json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

WORK_DIR = f"{OUTPUT_DIR}/magentic_code_runs"


async def main() -> None:
    model_client = OllamaChatCompletionClient(model="llama3.2")

    # Agents
    math_agent = AssistantAgent(
        "MathAgent",
        model_client=model_client,
        system_message="You are good at numeric computation and concise results."
    )

    proof_agent = AssistantAgent(
        "ProofAgent",
        model_client=model_client,
        system_message="You explain reasoning and proofs step by step."
    )

    coder_agent = AssistantAgent(
        "CoderAgent",
        model_client=model_client,
        system_message=(
            "You are a coder. When appropriate, write full runnable Python code inside ```python``` blocks. "
            "Make sure code is self-contained and uses print for output."
        )
    )

    # Executor with LocalCommandLineCodeExecutor
    work_dir = Path(WORK_DIR)
    work_dir.mkdir(exist_ok=True)

    local_executor = LocalCommandLineCodeExecutor(
        timeout=60,
        work_dir=work_dir,
        cleanup_temp_files=False,
        # virtual_env_context optionally passed
    )

    executor_agent = CodeExecutorAgent(
        "ExecutorAgent",
        model_client=model_client,
        code_executor=local_executor
    )

    # Build group chat with orchestrator
    team = MagenticOneGroupChat(
        participants=[math_agent, proof_agent, coder_agent, executor_agent],
        model_client=model_client,
        max_turns=10,
        max_stalls=2,
    )

    logger.info("\n=== Conversation Start ===\n")
    async for event in team.run_stream(task="Compute the sum of first 10 Fibonacci numbers in Python, print result, and prove the formula."):
        logger.gray("Event:")
        logger.info(format_json(event))
        # Log speaker and content if it's a chat message
        logger.info(f"Sender: {event.source}\nContent:{event.content}")
    logger.info("\n=== Conversation End ===\n")

    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
