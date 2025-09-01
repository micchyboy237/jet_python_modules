import asyncio
import logging
import os
from typing import Optional
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import RoundRobinGroupChat
from jet.libs.autogen.ollama_client import OllamaChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from pydantic import BaseModel

# Configure logging with progress tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("web_surfer.log")
    ]
)
logger = logging.getLogger("MultimodalWebSurferExample")


class ProgressTracker(BaseModel):
    total_steps: int
    current_step: int = 0
    current_task: Optional[str] = None

    def update(self, task: str, increment: bool = True) -> None:
        self.current_task = task
        if increment:
            self.current_step += 1
        logger.info(
            f"Progress: {self.current_step}/{self.total_steps} - {task}")


async def main() -> None:
    # Initialize progress tracker
    progress = ProgressTracker(total_steps=5)
    progress.update("Initializing web surfer agent")

    # Create debug and downloads directories
    debug_dir = "debug_screenshots"
    downloads_dir = "downloads"
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(downloads_dir, exist_ok=True)

    # Configure the Ollama model client
    model_client = OllamaChatCompletionClient(
        model="llama3.2",
        host="http://localhost:11434"
    )

    # Initialize MultimodalWebSurfer with all available arguments
    web_surfer = MultimodalWebSurfer(
        name="WebSurferDemo",
        model_client=model_client,
        downloads_folder=downloads_dir,
        description="A demo web surfer agent using Ollama for multimodal tasks.",
        debug_dir=debug_dir,
        headless=False,  # Show browser for demo
        start_page="https://www.github.com",
        animate_actions=True,
        to_save_screenshots=True,
        use_ocr=True,
        browser_channel="chrome",  # Use Chrome channel
        browser_data_dir="browser_data",
        to_resize_viewport=True
    )

    progress.update("Setting up team and task")

    # Define a team with the web surfer
    agent_team = RoundRobinGroupChat([web_surfer], max_turns=3)

    # Define a sample task
    task = "Navigate to the AutoGen README on GitHub and summarize it."

    progress.update("Running the team")
    try:
        # Run the team and stream messages to the console
        stream = agent_team.run_stream(task=task)
        await Console(stream)
        progress.update("Task completed successfully")
    except Exception as e:
        logger.error(f"Error during task execution: {str(e)}")
        progress.update(f"Task failed: {str(e)}", increment=False)
    finally:
        progress.update("Closing browser")
        await web_surfer.close()
        progress.update("Browser closed, execution complete")

if __name__ == "__main__":
    # Set event loop policy for Windows compatibility
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
