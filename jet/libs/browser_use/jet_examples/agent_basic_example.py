from browser_use.agent.service import Agent
from browser_use.llm.ollama.chat import ChatOllama
from browser_use.browser.session import BrowserSession, BrowserProfile
from browser_use.tools.service import Tools
from browser_use.llm.messages import ContentPartImageParam
from browser_use.agent.views import AgentState
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def basic_agent_example():
    """Demonstrates basic Agent usage with minimal arguments."""
    # Given: A simple task to navigate to a website
    task = "Navigate to https://example.com"
    
    # When: Creating an agent with minimal configuration
    agent = Agent(
        task=task,
        llm=ChatOllama(model="llama3.1"),  # Using Ollama LLM
    )
    
    # Then: Run the agent
    history = await agent.run(max_steps=3)
    logger.info(f"Task completed with {len(history.history)} steps")
    return history

if __name__ == "__main__":
    asyncio.run(basic_agent_example())