import pytest
import asyncio
from browser_use.agent.service import Agent
from browser_use.llm.ollama.chat import ChatOllama
from browser_use.browser.session import BrowserProfile
from browser_use.agent.views import AgentHistoryList
from jet.libs.browser_use.jet_examples.agent_basic_example import basic_agent_example
from jet.libs.browser_use.jet_examples.agent_advanced_example import advanced_agent_example

@pytest.mark.asyncio
async def test_basic_agent_example():
    """Test basic agent example functionality."""
    # Given: A simple task to navigate to a website
    expected_steps = 3
    expected_task = "Navigate to https://example.com"
    
    # When: Running the basic agent example
    history = await basic_agent_example()
    
    # Then: Verify the agent executed correctly
    result = history
    expected = AgentHistoryList(history=[], usage=None)
    assert isinstance(result, AgentHistoryList), "Result should be AgentHistoryList"
    assert len(result.history) <= expected_steps, f"Expected at most {expected_steps} steps"
    assert result.task == expected_task, f"Expected task {expected_task}"

@pytest.mark.asyncio
async def test_advanced_agent_example():
    """Test advanced agent example with all arguments."""
    # Given: A complex task with full configuration
    expected_steps = 5
    expected_task = "Search for 'AI news' on https://news.google.com and summarize findings"
    expected_model = "llama3.1"
    
    # When: Running the advanced agent example
    history = await advanced_agent_example()
    
    # Then: Verify the agent executed with correct configuration
    result = history
    expected = AgentHistoryList(history=[], usage=None)
    assert isinstance(result, AgentHistoryList), "Result should be AgentHistoryList"
    assert len(result.history) <= expected_steps, f"Expected at most {expected_steps} steps"
    assert result.task == expected_task, f"Expected task {expected_task}"
    assert result.model == expected_model, f"Expected model {expected_model}"

@pytest.fixture(autouse=True)
async def cleanup():
    """Cleanup after tests."""
    yield
    # Add any necessary cleanup here
    await asyncio.sleep(0)