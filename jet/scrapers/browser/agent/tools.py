"""
Playwright Tool Definitions and Registry
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseToolModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def get_tool_description(cls) -> str:
        return cls.__doc__ or "No description provided."


class NavigateArgs(BaseToolModel):
    """Navigate to a new URL. Use only for starting a new site."""

    url: str = Field(description="Absolute URL")


class ClickArgs(BaseToolModel):
    """
    Click an interactive element.
    RULE: Only use for buttons/links that advance the task.
    CAUTION: Do not use to clear inputs. If click fails, element may be off-screen.
    """

    target: str = Field(description="Element ref from snapshot (e.g., 'e39')")
    doubleClick: Optional[bool] = Field(False)
    button: Literal["left", "right", "middle"] = Field("left")


class TypeArgs(BaseToolModel):
    """
    Type into a SINGLE input field.
    RULE: If multiple fields need filling, use browser_fill_form instead.
    """

    target: str = Field(description="Element ref from snapshot")
    text: str = Field(description="Text to type")
    submit: Optional[bool] = Field(
        False, description="Press Enter after typing? Only if no Submit button exists."
    )


class FillFormArgs(BaseToolModel):
    """
    Fill MULTIPLE input fields at once.
    RULE: MANDATORY if 2 or more fields need updating in the current view.
    This is faster and more reliable than multiple browser_type calls.
    """

    class FormField(BaseModel):
        target: str = Field(description="Element ref from snapshot")
        value: str = Field(description="Exact value from User Goal")

    fields: List[FormField] = Field(description="List of targets and values to fill")


class WaitForArgs(BaseToolModel):
    """Wait for time or text visibility. Use after clicks that trigger loading."""

    time: Optional[float] = Field(None, description="Seconds to wait")
    text: Optional[str] = Field(None, description="Text to wait for")


class PressKeyArgs(BaseToolModel):
    """Press a key (Enter, ArrowDown). Use for navigation or submitting forms without buttons."""

    key: str = Field(description="Key name (e.g., 'Enter')")


class HoverArgs(BaseToolModel):
    """Hover to reveal hidden menus or tooltips."""

    target: str = Field(description="Element ref from snapshot")


class FindArgs(BaseToolModel):
    """Search text within the current snapshot. Does not interact with the page."""

    text: Optional[str] = Field(None)
    regex: Optional[str] = Field(None)
    maxResults: Optional[int] = Field(None)


class SelectOptionArgs(BaseToolModel):
    """Select from a native <select> dropdown. Do not use for custom comboboxes."""

    target: str = Field(description="Element ref of <select>")
    values: List[str] = Field(description="Option values to select")


class HandleDialogArgs(BaseToolModel):
    """Handle JS alerts/prompts. Must accept or dismiss to proceed."""

    accept: bool = Field(description="True to accept, False to dismiss")
    promptText: Optional[str] = Field(None)


class EvaluateArgs(BaseToolModel):
    """
    Execute JavaScript.
    RULE: LAST RESORT. Only use if data is hidden or not in snapshot.
    If data is visible in snapshot, extract it manually and use final_answer.
    """

    function: str = Field(description="JS function string")
    target: Optional[str] = Field(None)


class RunCodeArgs(BaseToolModel):
    """Execute arbitrary JS. Last resort for complex automation."""

    code: str = Field(description="JS code")


class ExtractContentArgs(BaseToolModel):
    """
    Extracts the main readable text content from the current page into a clean Markdown format.

    WHEN TO USE:
    1. COMPREHENSIVE LISTING: When the goal requires extracting a list of items (e.g., job titles, products, articles) that may span beyond the visible viewport or are scattered across the page.
    2. TOKEN EFFICIENCY: When the page is large and the standard snapshot contains too much UI noise (headers, footers, scripts). This tool strips away non-content elements.
    3. LINK EXTRACTION: When you need to capture many links. This tool automatically shortens long URLs to 'link_X' placeholders and provides a map, saving significant tokens.

    WHEN NOT TO USE:
    1. SIMPLE FACTS: If the answer is a single piece of text clearly visible in the current snapshot, just use 'final_answer'.
    2. INTERACTION: This tool is read-only. Do not use it if you need to click or type first.

    NOTE: The output is saved to 'markdown_content.md' and 'url_map.json' in the step directory.
    """

    pass


class FinalAnswerArgs(BaseToolModel):
    """
    Provide the final result.
    RULE: Call IMMEDIATELY when goal is met. Extract data from snapshot if present.
    """

    result: Any = Field(description="Final answer (string, list, or object)")


class AgentDecision(BaseModel):
    thought: str = Field(description="Brief reasoning based on Snapshot vs Goal")
    tool: str = Field(description="Tool name")
    arguments: Optional[Dict[str, Any]] = Field(None)


TOOLS_TRIGGERING_STATE_CHANGE = {
    "browser_navigate",
    "browser_navigate_back",
    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_press_key",
    "browser_hover",
    "browser_select_option",
    "browser_handle_dialog",
    "browser_run_code_unsafe",
    "browser_evaluate",
    "browser_wait_for",
}

TOOL_MODEL_MAP = {
    "browser_navigate": NavigateArgs,
    "browser_navigate_back": BaseToolModel,
    "browser_click": ClickArgs,
    "browser_type": TypeArgs,
    "browser_fill_form": FillFormArgs,
    "browser_wait_for": WaitForArgs,
    "browser_press_key": PressKeyArgs,
    "browser_hover": HoverArgs,
    # "browser_find": FindArgs,
    "browser_select_option": SelectOptionArgs,
    "browser_handle_dialog": HandleDialogArgs,
    "browser_evaluate": EvaluateArgs,
    "browser_run_code_unsafe": RunCodeArgs,
    "browser_extract_content": ExtractContentArgs,
    "browser_close": BaseToolModel,
    "final_answer": FinalAnswerArgs,
}


def get_tool_registry() -> Dict[str, Dict]:
    registry = {}
    for name, model_class in TOOL_MODEL_MAP.items():
        registry[name] = {
            "model": model_class,
            "desc": model_class.get_tool_description(),
        }
    return registry


def get_decision_schema() -> Dict[str, Any]:
    schema = AgentDecision.model_json_schema()
    return {"name": "AgentDecision", "strict": True, "schema": schema}
