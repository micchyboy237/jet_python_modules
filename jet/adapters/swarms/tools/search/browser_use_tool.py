import os
import asyncio

from dotenv import load_dotenv
from datetime import datetime
from browser_use import Agent, ChatOpenAI

from jet.llm.mlx.adapters.mlx_langchain_llm_adapter import ChatMLX
from jet.models.model_types import LLMModelType
from jet.logger import CustomLogger

# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
filename_no_ext = os.path.splitext(os.path.basename(__file__))[0]
log_file = os.path.join(log_dir, f"{filename_no_ext}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = CustomLogger(log_file, overwrite=True)

load_dotenv()


def browser_agent(task: str, model_name: LLMModelType = "mlx-community/Llama-3.2-3B-Instruct-4bit"):
    """
    Executes a browser-based agent to perform a specified task using a language model.

    This function sets up and runs an asynchronous agent that utilizes a language model
    (default: "mlx-community/Llama-3.2-3B-Instruct-4bit") to complete the provided task. The agent is executed in an
    asyncio event loop, and its output is printed to the console.

    Args:
        task (str):
            A description of the task for the agent to perform. This should be a clear,
            concise instruction or query that the agent can act upon using browser-based tools.
        model_name (str, optional):
            The name of the language model to use for the agent. Defaults to "mlx-community/Llama-3.2-3B-Instruct-4bit".
            This parameter allows you to specify different models as needed.

    Returns:
        None. The function prints the agent's output to the console.

    Example:
        >>> browser_agent("Find the number of stars of the swarms")
        # Output will be printed to the console.

    Notes:
        - The function uses asyncio to run the agent asynchronously.
        - The agent's output is processed with `model_dump()` before being printed.
        - Requires the `Agent` and `ChatOpenAI` classes from the `browser_use` module.
        - Assumes that environment variables (such as API keys) are loaded via dotenv.
    """

    async def run_agent():
        """
        Asynchronously creates and runs the Agent to perform the specified task.

        Returns:
            The result of the agent's run method, which contains the output of the task.
        """
        agent = Agent(
            task=task,
            llm=ChatMLX(model=model_name),
        )
        return await agent.run()

    # Run the asynchronous agent and obtain the output.
    out = asyncio.run(run_agent())
    # Process the output with model_dump (for serialization or inspection).
    out.model_dump()
    # Print the final output to the console.
    print(out)


if __name__ == "__main__":
    # Example usage: instruct the agent to find the number of stars of the swarms.
    browser_agent("Find the number of stars of the swarms")
