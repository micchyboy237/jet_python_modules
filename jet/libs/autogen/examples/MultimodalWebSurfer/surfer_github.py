import os
import shutil
import asyncio
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from jet.logger import CustomLogger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


async def main() -> None:
    # Initialize the agent with Chrome and 80% zoom
    web_surfer_agent = MultimodalWebSurfer(
        name="MultimodalWebSurfer",
        model_client=OllamaChatCompletionClient(model="llama3.2"),
        browser_channel="chrome",
        headless=False,
        animate_actions=False,
        to_save_screenshots=True,
        debug_dir=f"{OUTPUT_DIR}/debug_screens",
        browser_data_dir=f"{OUTPUT_DIR}/browser_data_dir",
    )

    # Define a team
    agent_team = RoundRobinGroupChat([web_surfer_agent], max_turns=5)

    # Task to locate the search bar's target_id and use it
    task = (
        "1. Navigate to http://jethros-macbook-air.local:3000/search.\n"
        "2. Wait for the page to load fully.\n"
        "3. Identify the target_id of the search bar.\n"
        "4. Type 'AutoGen GitHub' into the search bar."
    )

    # Run the team and stream messages to the console
    stream = agent_team.run_stream(task=task)
    await Console(stream)
    # Close the browser
    await web_surfer_agent.close()

if __name__ == "__main__":
    asyncio.run(main())
