import os
import shutil
import asyncio
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient

from jet.logger import CustomLogger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


async def main():
    # Initialize the model client (e.g., OpenAI GPT-4o)
    model_client = OllamaChatCompletionClient(model="llama3.2")

    # Define the web surfer agent with screenshot saving enabled
    web_surfer_agent = MultimodalWebSurfer(
        name="WebSurfer",
        model_client=model_client,
        to_save_screenshots=True,  # Enable screenshot saving
        debug_dir="./screenshots",  # Directory to save screenshots
        headless=True,  # Run browser in headless mode
        start_page="http://jethros-macbook-air.local:8888",  # Starting URL
    )

    # Define the task
    task = """
    Navigate to https://github.com.
    Find the search bar and input the query 'AutoGen Microsoft'.
    Submit the search.
    Extract the text content and links from the search results page.
    Save screenshots of the search results page.
    """

    # Define a team with the web surfer agent
    agent_team = RoundRobinGroupChat([web_surfer_agent], max_turns=5)

    # Run the task and stream messages to the console
    stream = agent_team.run_stream(task=task)
    await Console(stream)

    # Close the browser
    await web_surfer_agent.close()

if __name__ == "__main__":
    asyncio.run(main())
