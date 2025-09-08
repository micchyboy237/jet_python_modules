from browser_use.agent.service import Agent
from browser_use.llm.ollama.chat import ChatOllama
from browser_use.browser.session import BrowserProfile
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def basic_agent_example():
    """Demonstrates basic Agent usage with minimal arguments and window size."""
    # Given: A simple task to navigate to a website
    task = "Navigate to https://example.com"
    
    # When: Creating an agent with minimal configuration and custom browser size
    browser_profile = BrowserProfile(
        window_size={"width": 1440, "height": 900}  # Set browser window size to 1440x900 pixels
    )
    agent = Agent(
        task=task,
        llm=ChatOllama(model="llama3.2"),
        browser_profile=browser_profile
    )
    
    # Then: Run the agent
    history = await agent.run(max_steps=3)
    logger.info(f"Task completed with {len(history.history)} steps")
    return history

if __name__ == "__main__":
    asyncio.run(basic_agent_example())