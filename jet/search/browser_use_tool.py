import os
import shutil
import asyncio

from dotenv import load_dotenv
from browser_use import Agent, BrowserProfile

from jet.adapters.browser_use.ollama.chat import ChatOllama
from jet.logger import logger

load_dotenv()

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")


def browser_agent(task: str, model_name: str = "llama3.2"):
    """
    Executes a browser-based agent to perform a specified task using a language model.

    This function sets up and runs an asynchronous agent that utilizes a language model
    (default: "llama3.2") to complete the provided task. The agent is executed in an
    asyncio event loop, and its output is printed to the console.

    Args:
        task (str):
            A description of the task for the agent to perform. This should be a clear,
            concise instruction or query that the agent can act upon using browser-based tools.
        model_name (str, optional):
            The name of the language model to use for the agent. Defaults to "llama3.2".
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
        browser_profile = BrowserProfile(headless=True)
        agent = Agent(
            task=task,
            llm=ChatOllama(model=model_name),
            browser_profile=browser_profile,
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
