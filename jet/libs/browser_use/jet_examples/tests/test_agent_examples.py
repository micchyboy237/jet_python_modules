import pytest
import asyncio
from browser_use.agent.service import Agent
from browser_use.llm.ollama.chat import ChatOllama
from browser_use.browser.session import BrowserProfile
from browser_use.agent.views import AgentHistoryList
from jet.libs.browser_use.jet_examples.agent_basic_example import basic_agent_example
from jet.libs.browser_use.jet_examples.agent_advanced_example import advanced_agent_example
from jet.libs.browser_use.jet_examples.agent_callbacks_example import callbacks_agent_example
from jet.libs.browser_use.jet_examples.agent_history_replay_example import history_replay_agent_example
from jet.libs.browser_use.jet_examples.agent_cloud_sync_example import cloud_sync_agent_example

@pytest.mark.asyncio
async def test_basic_agent_example():
    """Test basic agent example functionality."""
    # Given: A simple task to navigate to a website
    expected_steps = 3
    expected_task = "Navigate to https://example.com"
    expected_window_size = {"width": 1440, "height": 900}
    
    # When: Running the basic agent example
    history = await basic_agent_example()
    
    # Then: Verify the agent executed correctly
    result = history
    expected = AgentHistoryList(history=[], usage=None)
    assert isinstance(result, AgentHistoryList), "Result should be AgentHistoryList"
    assert len(result.history) <= expected_steps, f"Expected at most {expected_steps} steps"
    assert result.task == expected_task, f"Expected task {expected_task}"
    assert result.browser_profile.window_size == expected_window_size, f"Expected window size {expected_window_size}"

@pytest.mark.asyncio
async def test_advanced_agent_example():
    """Test advanced agent example with all arguments."""
    # Given: A complex task with full configuration
    expected_steps = 5
    expected_task = "Search for 'AI news' on https://news.google.com and summarize findings"
    expected_model = "llama3.2"
    expected_window_size = {"width": 1440, "height": 900}
    
    # When: Running the advanced agent example
    history = await advanced_agent_example()
    
    # Then: Verify the agent executed with correct configuration
    result = history
    expected = AgentHistoryList(history=[], usage=None)
    assert isinstance(result, AgentHistoryList), "Result should be AgentHistoryList"
    assert len(result.history) <= expected_steps, f"Expected at most {expected_steps} steps"
    assert result.task == expected_task, f"Expected task {expected_task}"
    assert result.model == expected_model, f"Expected model {expected_model}"
    assert result.browser_profile.window_size == expected_window_size, f"Expected window size {expected_window_size}"

@pytest.mark.asyncio
async def test_callbacks_agent_example():
    """Test agent example with callbacks."""
    # Given: A task to demonstrate callbacks
    expected_steps = 3
    expected_task = "Visit https://wikipedia.org"
    expected_window_size = {"width": 1440, "height": 900}
    
    # When: Running the callbacks agent example
    history = await callbacks_agent_example()
    
    # Then: Verify the agent executed with callbacks
    result = history
    expected = AgentHistoryList(history=[], usage=None)
    assert isinstance(result, AgentHistoryList), "Result should be AgentHistoryList"
    assert len(result.history) <= expected_steps, f"Expected at most {expected_steps} steps"
    assert result.task == expected_task, f"Expected task {expected_task}"
    assert result.browser_profile.window_size == expected_window_size, f"Expected window size {expected_window_size}"

@pytest.mark.asyncio
async def test_history_replay_agent_example():
    """Test agent example with history replay."""
    # Given: A task to demonstrate history replay
    expected_task = "Navigate to https://example.com"
    expected_window_size = {"width": 1440, "height": 900}
    
    # When: Running the history replay agent example
    results = await history_replay_agent_example()
    
    # Then: Verify the agent executed and replayed history
    assert isinstance(results, list), "Result should be a list of ActionResult"
    assert results, "Expected non-empty results from history replay"
    assert results[0].browser_profile.window_size == expected_window_size, f"Expected window size {expected_window_size}"

@pytest.mark.asyncio
async def test_cloud_sync_agent_example():
    """Test agent example with cloud sync."""
    # Given: A task to demonstrate cloud sync
    expected_steps = 3
    expected_task = "Visit https://example.com and take a screenshot"
    expected_window_size = {"width": 1440, "height": 900}
    
    # When: Running the cloud sync agent example
    history = await cloud_sync_agent_example()
    
    # Then: Verify the agent executed with cloud sync
    result = history
    expected = AgentHistoryList(history=[], usage=None)
    assert isinstance(result, AgentHistoryList), "Result should be AgentHistoryList"
    assert len(result.history) <= expected_steps, f"Expected at most {expected_steps} steps"
    assert result.task == expected_task, f"Expected task {expected_task}"
    assert result.browser_profile.window_size == expected_window_size, f"Expected window size {expected_window_size}"
    
@pytest.fixture(autouse=True)
async def cleanup():
    """Cleanup after tests."""
    yield
    await asyncio.sleep(0)