"""
Playwright Tool Definitions and Registry
Contains all Pydantic models, tool descriptions, and execution logic.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class NavigateArgs(BaseModel):
    url: str = Field(description="The URL to navigate to")


class ClickArgs(BaseModel):
    target: str = Field(description="Target element ref (e.g., e39) or selector")
    doubleClick: Optional[bool] = Field(False, description="Whether to double click")
    button: Literal["left", "right", "middle"] = Field(
        "left", description="Mouse button to use"
    )


class TypeArgs(BaseModel):
    target: str = Field(description="Target element ref (e.g., e39) or selector")
    text: str = Field(description="Text to type")
    submit: Optional[bool] = Field(
        False, description="Whether to press Enter after typing"
    )


class FillFormArgs(BaseModel):
    class FormField(BaseModel):
        target: str = Field(
            description="Target element ref (e.g., e39) or selector for the input field"
        )
        value: str = Field(description="Value to fill into the field")

    fields: List[FormField] = Field(
        description="List of fields to fill in the form. Each field must have a 'target' and a 'value'."
    )


class WaitForArgs(BaseModel):
    time: Optional[float] = Field(None, description="Time to wait in seconds")
    text: Optional[str] = Field(None, description="Text to wait for")


class PressKeyArgs(BaseModel):
    key: str = Field(description="Name of the key to press")


class HoverArgs(BaseModel):
    target: str = Field(description="Target element ref (e.g., e39) or selector")


class FindArgs(BaseModel):
    text: Optional[str] = Field(None, description="Plain text to search for")
    regex: Optional[str] = Field(None, description="Regular expression to search for")
    maxResults: Optional[int] = Field(None, description="Max results")


class SelectOptionArgs(BaseModel):
    target: str = Field(description="Target element ref")
    values: List[str] = Field(description="Values to select")


class HandleDialogArgs(BaseModel):
    accept: bool = Field(description="Whether to accept the dialog")
    promptText: Optional[str] = Field(None, description="Prompt text")


class EvaluateArgs(BaseModel):
    function: str = Field(description="JS function to evaluate")
    target: Optional[str] = Field(None, description="Target element ref")


class RunCodeArgs(BaseModel):
    code: str = Field(description="JS code to execute")


class FinalAnswerArgs(BaseModel):
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


def get_tool_registry() -> Dict[str, Dict]:
    """Returns the complete tool registry with models and descriptions."""
    return {
        "browser_navigate": {
            "model": NavigateArgs,
            "desc": "Navigate to a new URL. Use this only when you need to go to a completely different page or start over.",
        },
        "browser_navigate_back": {
            "model": BaseModel,
            "desc": "Go back to the previous page in browser history. Use this if you clicked a link by mistake or need to return to search results.",
        },
        "browser_click": {
            "model": ClickArgs,
            "desc": "Click an element like a button, link, or checkbox. Use the 'target' ref from the snapshot. Do NOT use this for text inputs; use browser_type instead.",
        },
        "browser_type": {
            "model": TypeArgs,
            "desc": "Type text into a single input field (textbox, textarea, or combobox). Use this for filling one field at a time. Set 'submit' to true if you need to press Enter after typing.",
        },
        "browser_fill_form": {
            "model": FillFormArgs,
            "desc": "Efficiently fill multiple input fields at once. Use this when you need to type into several fields (e.g., username and password) in one step. Provide a list of {'target': ref, 'value': text} objects.",
        },
        "browser_wait_for": {
            "model": WaitForArgs,
            "desc": "Wait for a specific amount of time or for specific text to appear on the page. Use this after clicking a search button or loading content to ensure elements are visible before interacting.",
        },
        "browser_press_key": {
            "model": PressKeyArgs,
            "desc": "Press a specific key (e.g., 'Enter', 'ArrowDown', 'Escape'). Use this for keyboard navigation or submitting forms without a visible button.",
        },
        "browser_hover": {
            "model": HoverArgs,
            "desc": "Hover the mouse over an element. Use this to reveal hidden menus, tooltips, or dropdowns that only appear on hover.",
        },
        "browser_find": {
            "model": FindArgs,
            "desc": "Search for specific text within the current snapshot. Use this to quickly locate information without scanning the entire snapshot manually.",
        },
        "browser_select_option": {
            "model": SelectOptionArgs,
            "desc": "Select one or more options from a <select> dropdown menu. Use this when the target is a native dropdown, not a custom combobox.",
        },
        "browser_handle_dialog": {
            "model": HandleDialogArgs,
            "desc": "Handle JavaScript dialogs like alerts, confirms, or prompts. Use this immediately if a dialog blocks interaction with the page.",
        },
        "browser_evaluate": {
            "model": EvaluateArgs,
            "desc": "Execute JavaScript in the page context and return the result. Use this for complex data extraction or interactions that standard tools cannot handle.",
        },
        "browser_run_code_unsafe": {
            "model": RunCodeArgs,
            "desc": "Execute arbitrary JavaScript code in the page. Use this as a last resort for advanced automation tasks.",
        },
        "browser_close": {
            "model": BaseModel,
            "desc": "Close the browser. Use this ONLY when the task is completely finished and no further actions are needed.",
        },
        "final_answer": {
            "model": FinalAnswerArgs,
            "desc": "Provide the final result of the task. Call this IMMEDIATELY once you have the required information. The 'result' can be a string, JSON object, or array depending on the goal.",
        },
    }


def get_decision_schema() -> Dict[str, Any]:
    """Returns the JSON schema for agent decisions."""
    schema = AgentDecision.model_json_schema()
    return {
        "name": "AgentDecision",
        "strict": True,
        "schema": schema,
    }
