"""
Playwright Tool Definitions and Registry
Contains all Pydantic models, tool descriptions, and execution logic.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseToolModel(BaseModel):
    """Base class for all tool models to enforce consistent configuration."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def get_tool_description(cls) -> str:
        """Returns the description of the tool. Override in subclasses."""
        return cls.__doc__ or "No description provided."


class NavigateArgs(BaseToolModel):
    """Navigate to a new URL. Use this only when you need to go to a completely different page or start over."""

    url: str = Field(description="The URL to navigate to")


class ClickArgs(BaseToolModel):
    """Click an element like a button, link, or checkbox. Use the 'target' ref from the snapshot. Do NOT use this for text inputs; use browser_type instead."""

    target: str = Field(description="Target element ref (e.g., e39) or selector")
    doubleClick: Optional[bool] = Field(False, description="Whether to double click")
    button: Literal["left", "right", "middle"] = Field(
        "left", description="Mouse button to use"
    )


class TypeArgs(BaseToolModel):
    """Type text into a single input field (textbox, textarea, or combobox). Use this for filling one field at a time. Set 'submit' to true if you need to press Enter after typing."""

    target: str = Field(description="Target element ref (e.g., e39) or selector")
    text: str = Field(description="Text to type")
    submit: Optional[bool] = Field(
        False, description="Whether to press Enter after typing"
    )


class FillFormArgs(BaseToolModel):
    """Efficiently fill multiple input fields at once. Use this when you need to type into several fields (e.g., username and password) in one step. Provide a list of {'target': ref, 'value': text} objects."""

    class FormField(BaseModel):
        target: str = Field(
            description="Target element ref (e.g., e39) or selector for the input field"
        )
        value: str = Field(description="Value to fill into the field")

    fields: List[FormField] = Field(
        description="List of fields to fill in the form. Each field must have a 'target' and a 'value'."
    )


class WaitForArgs(BaseToolModel):
    """Wait for a specific amount of time or for specific text to appear on the page. Use this after clicking a search button or loading content to ensure elements are visible before interacting."""

    time: Optional[float] = Field(None, description="Time to wait in seconds")
    text: Optional[str] = Field(None, description="Text to wait for")


class PressKeyArgs(BaseToolModel):
    """Press a specific key (e.g., 'Enter', 'ArrowDown', 'Escape'). Use this for keyboard navigation or submitting forms without a visible button."""

    key: str = Field(description="Name of the key to press")


class HoverArgs(BaseToolModel):
    """Hover the mouse over an element. Use this to reveal hidden menus, tooltips, or dropdowns that only appear on hover."""

    target: str = Field(description="Target element ref (e.g., e39) or selector")


class FindArgs(BaseToolModel):
    """Search for specific text within the current snapshot. Use this to quickly locate information without scanning the entire snapshot manually."""

    text: Optional[str] = Field(None, description="Plain text to search for")
    regex: Optional[str] = Field(None, description="Regular expression to search for")
    maxResults: Optional[int] = Field(None, description="Max results")


class SelectOptionArgs(BaseToolModel):
    """Select one or more options from a <select> dropdown menu. Use this when the target is a native dropdown, not a custom combobox."""

    target: str = Field(description="Target element ref")
    values: List[str] = Field(description="Values to select")


class HandleDialogArgs(BaseToolModel):
    """Handle JavaScript dialogs like alerts, confirms, or prompts. Use this immediately if a dialog blocks interaction with the page."""

    accept: bool = Field(description="Whether to accept the dialog")
    promptText: Optional[str] = Field(None, description="Prompt text")


class EvaluateArgs(BaseToolModel):
    """Execute JavaScript in the page context and return the result. Use this for complex data extraction or interactions that standard tools cannot handle."""

    function: str = Field(description="JS function to evaluate")
    target: Optional[str] = Field(None, description="Target element ref")


class RunCodeArgs(BaseToolModel):
    """Execute arbitrary JavaScript code in the page. Use this as a last resort for advanced automation tasks."""

    code: str = Field(description="JS code to execute")


class FinalAnswerArgs(BaseToolModel):
    """Provide the final result of the task. Call this IMMEDIATELY once you have the required information. The 'result' can be a string, JSON object, or array depending on the goal."""

    result: Any = Field(
        description="The final answer. Can be a string, JSON object, array, or number."
    )


class AgentDecision(BaseModel):
    thought: str = Field(description="Brief reasoning")
    tool: str = Field(description="Name of the tool to execute")
    arguments: Optional[Dict[str, Any]] = Field(
        None, description="Arguments for the tool"
    )
    finished: bool = Field(False, description="Whether the goal has been achieved")


TOOLS_RETURNING_SNAPSHOTS = {
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
}

TOOLS_NOT_RETURNING_SNAPSHOTS = {
    "browser_wait_for",
    "browser_find",
    "browser_evaluate",
    "browser_close",
    "final_answer",
}

# Mapping of tool names to their models
TOOL_MODEL_MAP = {
    "browser_navigate": NavigateArgs,
    "browser_navigate_back": BaseToolModel,  # No args
    "browser_click": ClickArgs,
    "browser_type": TypeArgs,
    "browser_fill_form": FillFormArgs,
    "browser_wait_for": WaitForArgs,
    "browser_press_key": PressKeyArgs,
    "browser_hover": HoverArgs,
    "browser_find": FindArgs,
    "browser_select_option": SelectOptionArgs,
    "browser_handle_dialog": HandleDialogArgs,
    "browser_evaluate": EvaluateArgs,
    "browser_run_code_unsafe": RunCodeArgs,
    "browser_close": BaseToolModel,  # No args
    "final_answer": FinalAnswerArgs,
}


def get_tool_registry() -> Dict[str, Dict]:
    """Returns the complete tool registry with models and descriptions extracted from models."""
    registry = {}
    for name, model_class in TOOL_MODEL_MAP.items():
        registry[name] = {
            "model": model_class,
            "desc": model_class.get_tool_description(),
        }
    return registry


def get_decision_schema() -> Dict[str, Any]:
    """Returns the JSON schema for agent decisions."""
    schema = AgentDecision.model_json_schema()
    return {
        "name": "AgentDecision",
        "strict": True,
        "schema": schema,
    }
