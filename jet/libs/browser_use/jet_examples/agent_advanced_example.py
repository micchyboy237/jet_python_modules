from browser_use.agent.service import Agent
from browser_use.llm.ollama.chat import ChatOllama
from browser_use.browser.session import BrowserSession, BrowserProfile
from browser_use.tools.service import Tools
from browser_use.llm.messages import ContentPartImageParam, ContentPartTextParam
from browser_use.agent.views import AgentState, AgentStepInfo
from pydantic import BaseModel
import asyncio
import logging
from typing import Any, Awaitable
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomOutput(BaseModel):
    result: str

async def step_start_callback(agent: Agent) -> None:
    logger.info(f"Starting step {agent.state.n_steps}")

async def step_end_callback(agent: Agent) -> None:
    logger.info(f"Finished step {agent.state.n_steps}")

async def error_check_callback() -> bool:
    return False  # Simulate no external errors

```python
async def advanced_agent_example():
    """Demonstrates Agent usage with all possible arguments and window size."""
    # Given: A complex task with multiple configurations
    task = "Search for 'AI news' on https://news.google.com and summarize findings"
    
    # When: Creating an agent with all possible arguments
    browser_profile = BrowserProfile(
        allowed_domains=["*.google.com"],
        downloads_path="/tmp/downloads",
        keep_alive=True,
        window_size={"width": 1440, "height": 900}  # Set browser window size to 1440x900 pixels
    )
```
    
    browser_session = BrowserSession(
        browser_profile=browser_profile,
        id="custom-session-123"
    )
    
    tools = Tools(display_files_in_done_text=True)
    
    sensitive_data = {
        "google.com": {"username": "user", "password": "pass"},
        "api_key": "xyz123"
    }
    
    initial_actions = [
        {"go_to_url": {"url": "https://news.google.com", "new_tab": False}},
        {"type": {"index": 1, "value": "AI news", "is_submit": True}}
    ]
    
    sample_images = [
        ContentPartTextParam(text="Example search page"),
        ContentPartImageParam(image_url="data:image/png;base64,iVBORw0KGgo...")
    ]
    
    agent = Agent(
        task=task,
        llm=ChatOllama(model="llama3.1", host="http://localhost:11434"),
        browser_profile=browser_profile,
        browser_session=browser_session,
        tools=tools,
        controller=tools,
        sensitive_data=sensitive_data,
        initial_actions=initial_actions,
        register_new_step_callback=step_start_callback,
        register_done_callback=lambda history: logger.info(f"Task done with {len(history.history)} steps"),
        register_external_agent_status_raise_error_callback=error_check_callback,
        output_model_schema=CustomOutput,
        use_vision=True,
        save_conversation_path="/tmp/conversations",
        save_conversation_path_encoding="utf-8",
        max_failures=5,
        override_system_message="Custom system prompt",
        extend_system_message="Additional system instructions",
        generate_gif="/tmp/agent_run.gif",
        available_file_paths=["/tmp/file1.txt", "/tmp/file2.txt"],
        include_attributes=["id", "class"],
        max_actions_per_step=15,
        use_thinking=True,
        flash_mode=False,
        max_history_items=100,
        page_extraction_llm=ChatOllama(model="llama3.1"),
        injected_agent_state=AgentState(),
        source="custom_source",
        file_system_path="/tmp/agent_files",
        task_id="custom_task_001",
        cloud_sync=None,
        calculate_cost=True,
        display_files_in_done_text=True,
        include_tool_call_examples=True,
        vision_detail_level="high",
        llm_timeout=120,
        step_timeout=180,
        directly_open_url=True,
        include_recent_events=True,
        sample_images=sample_images,
        final_response_after_failure=True,
        _url_shortening_limit=30
    )
    
    # Then: Run the agent
    history = await agent.run(max_steps=5, on_step_start=step_start_callback, on_step_end=step_end_callback)
    logger.info(f"Task completed with {len(history.history)} steps")
    return history

if __name__ == "__main__":
    asyncio.run(advanced_agent_example())