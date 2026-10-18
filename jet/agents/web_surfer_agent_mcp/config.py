import logging

from jet.adapters.llama_cpp.config import LLM_MODEL

# Default Constants
DEFAULT_MCP_SERVER_URL = "http://localhost:8931/mcp"
DEFAULT_START_URL = "https://demo.playwright.dev/todomvc"
DEFAULT_MAX_STEPS = 15
DEFAULT_MAX_REPEATS = 3
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TOP_P = 0.95
DEFAULT_PRESENCE_PENALTY = 1.5
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_TRACE_NAME = "web-surfer-agent"
DEFAULT_LLM_MODEL = LLM_MODEL

# Allowed Tools and Safe Optional Arguments
ALLOWED_TOOLS = {
    "browser_navigate",
    "browser_navigate_back",
    "browser_type",
    "browser_press_key",
    "browser_hover",
    "browser_find",
    "browser_click",
    "browser_fill_form",
    "browser_select_option",
    "browser_handle_dialog",
    "browser_evaluate",
    "browser_run_code_unsafe",
    "browser_snapshot",
    "browser_take_screenshot",
    "browser_wait_for",
    "browser_close",
}

SAFE_OPTIONAL = {"browser_type": {"submit"}}


def setup_logging(log_level: str):
    """Configure logging based on parsed arguments."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("mcp").setLevel(logging.WARNING)
