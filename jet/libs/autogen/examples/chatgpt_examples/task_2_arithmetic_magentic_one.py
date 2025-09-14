import os
import shutil
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient
from jet.logger import logger
from jet.transformers.formatters import format_json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")


async def main() -> None:
    model_client = OllamaChatCompletionClient(model="llama3.2")

    math_agent = AssistantAgent(
        "MathAgent",
        model_client=model_client,
        system_message="You are good at calculations and quick answers."
    )

    proof_agent = AssistantAgent(
        "ProofAgent",
        model_client=model_client,
        system_message="You like giving detailed proofs and logical reasoning."
    )

    team = MagenticOneGroupChat(
        participants=[math_agent, proof_agent],
        model_client=model_client,
        max_turns=6,
        max_stalls=2,
    )

    print("\n=== Conversation Start ===\n")
    async for event in team.run_stream(task="Find the sum of the first 10 integers and prove the formula works."):
        logger.gray("Event:")
        logger.info(format_json(event))
        # Log speaker and content if it's a chat message
        logger.info(f"Sender: {event.source}\nContent:{event.content}")

    print("\n=== Conversation End ===\n")


if __name__ == "__main__":
    asyncio.run(main())
