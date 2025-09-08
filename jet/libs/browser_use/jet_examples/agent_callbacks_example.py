from browser_use.agent.service import Agent
from browser_use.llm.ollama.chat import ChatOllama
from browser_use.browser.session import BrowserProfile
import asyncio
import logging
from typing import Awaitable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def new_step_callback(browser_state, model_output, step_number: int) -> None:
    """Callback for each new step."""
    logger.info(f"Step {step_number}: URL={browser_state.url}")

async def done_callback(history) -> None:
    """Callback when agent is done."""
    logger.info(f"Task completed with {len(history.history)} steps")

async def error_check_callback() -> bool:
    """Callback to check for external errors."""
    return False

async def callbacks_agent_example():
    """Demonstrates Agent usage with callback arguments."""
    # Given: A task to demonstrate callbacks
    task = "Visit https://wikipedia.org"
    
    # When: Creating an agent with callback configurations
    browser_profile = BrowserProfile(
        window_size=(1440, 900)  # Set browser window size
    )
    agent = Agent(
        task=task,
        llm=ChatOllama(model="llama3.2"),
        browser_profile=browser_profile,
        register_new_step_callback=new_step_callback,
        register_done_callback=done_callback,
        register_external_agent_status_raise_error_callback=error_check_callback
    )
    
    # Then: Run the agent
    history = await agent.run(max_steps=3)
    logger.info(f"Task completed with {len(history.history)} steps")
    return history

if __name__ == "__main__":
    asyncio.run(callbacks_agent_example())