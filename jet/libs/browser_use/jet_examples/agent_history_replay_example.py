from browser_use.agent.service import Agent
from browser_use.llm.ollama.chat import ChatOllama
from browser_use.browser.session import BrowserProfile
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def history_replay_agent_example():
    """Demonstrates Agent usage with history replay and file system."""
    # Given: A task to demonstrate history replay
    task = "Navigate to https://example.com"
    history_file = "/tmp/agent_history.json"
    
    # When: Creating an agent with history-related configurations
    browser_profile = BrowserProfile(
        window_size={"width": 1366, "height": 768}  # Set browser window size
    )
    agent = Agent(
        task=task,
        llm=ChatOllama(model="llama3.1"),
        browser_profile=browser_profile,
        file_system_path="/tmp/agent_files",
        save_conversation_path="/tmp/conversations"
    )
    
    # Then: Run the agent and save history
    history = await agent.run(max_steps=3)
    agent.save_history(history_file)
    logger.info(f"History saved to {history_file}")
    
    # Replay history
    results = await agent.load_and_rerun(history_file=history_file, max_retries=2, delay_between_actions=1.0)
    logger.info(f"History replay completed with {len(results)} results")
    return results

if __name__ == "__main__":
    asyncio.run(history_replay_agent_example())