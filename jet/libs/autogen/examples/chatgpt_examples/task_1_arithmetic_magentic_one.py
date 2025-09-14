import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient


async def main() -> None:
    # Use any OpenAI-compatible model, you can swap to your mlx_client later
    model_client = OllamaChatCompletionClient(model="llama3.2")

    # Two assistant agents with different roles
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

    # Group chat with MagenticOne orchestrator
    team = MagenticOneGroupChat(
        participants=[math_agent, proof_agent],
        model_client=model_client,
        max_turns=6,   # limit for demo
        max_stalls=2,
    )

    # Run with a sample task
    await Console(team.run_stream(task="Find the sum of the first 10 integers and prove the formula works."))


if __name__ == "__main__":
    asyncio.run(main())
