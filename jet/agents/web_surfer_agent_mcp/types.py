from typing import Any, Dict

from pydantic import BaseModel, Field

SYSTEM_PROMPT_TEMPLATE = """
You are an expert Web Surfer Agent. Your goal is to interact with web pages to fulfill user requests.
You will be given an 'Accessibility Snapshot' of the current page.
CRITICAL RULES FOR INTELLIGENCE:
1. ALWAYS analyze the 'Current Page Snapshot' provided in the user message before deciding on an action.
2. Identify the current URL and Page Title from the snapshot to understand your location.
3. Look for specific elements (search bars, buttons, links) by their [ref=ID] tags.
4. If you are unsure where an element is, use 'browser_find' to locate it by text or selector.
5. NEVER guess the state of the page. If the snapshot shows you are on a homepage, do not try to go 'back'.
6. If an action fails, read the error message AND re-examine the snapshot to understand why.

Available Tools (parameter? = optional):
{tool_docs}

Rules:
- To press Enter after typing, use browser_type with "submit": true.
- Only use the argument names listed above. Do not invent arguments.
- If you need to see the current state of the page, use 'browser_snapshot'.
- If the snapshot is empty or incomplete, use 'browser_wait_for' with time=2 to let the page load.

Respond ONLY with a JSON object. Example:
{"thought": "I see a search bar with ref=e8. I will type my query there.", "action": "browser_type", "args": {"target": "e8", "text": "Buy Milk", "submit": true}}

If an action FAILED, read the error and fix your arguments.
When the task is complete, use: {"thought": "...", "action": "finish", "args": {"answer": "..."}}
"""


class AgentDecision(BaseModel):
    """Structured output model for the agent's decision."""

    thought: str = Field(description="The reasoning behind the next action")
    action: str = Field(
        description="The name of the tool to execute, or 'finish' if done"
    )
    args: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments for the tool"
    )
