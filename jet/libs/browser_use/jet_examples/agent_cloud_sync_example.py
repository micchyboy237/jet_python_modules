from browser_use.agent.service import Agent
from browser_use.llm.ollama.chat import ChatOllama
from browser_use.browser.session import BrowserProfile
from browser_use.sync import CloudSync
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def cloud_sync_agent_example():
    """Demonstrates Agent usage with cloud sync and authentication."""
    # Given: A task to demonstrate cloud sync
    task = "Visit https://example.com and take a screenshot"
    
    # When: Creating an agent with cloud sync
    browser_profile = BrowserProfile(
        window_size={"width": 1440, "height": 900}  # Set browser window size
    )
    cloud_sync = CloudSync()
    agent = Agent(
        task=task,
        llm=ChatOllama(model="llama3.2"),
        browser_profile=browser_profile,
        cloud_sync=cloud_sync,
        generate_gif=True
    )
    
    # Authenticate cloud sync
    auth_success = await agent.authenticate_cloud_sync(show_instructions=True)
    logger.info(f"Cloud sync authentication: {'successful' if auth_success else 'failed'}")
    
    # Then: Run the agent
    history = await agent.run(max_steps=3)
    logger.info(f"Task completed with {len(history.history)} steps")
    return history

if __name__ == "__main__":
    asyncio.run(cloud_sync_agent_example())