import sys

from agent_framework import MCPStdioTool
from jet.adapters.llama_cpp.config import PHOENIX_BASE_URL
from jet_telemetry import initialize_telemetry

# Initialize telemetry
initialize_telemetry(service_name="web-surfer-agent-mcp", endpoint=PHOENIX_BASE_URL)


def get_mcp_browser_tool() -> MCPStdioTool:
    """
    Returns an MCPStdioTool configured to connect to a browser automation MCP server.

    In a production environment, replace 'command' and 'args' with your actual
    MCP server executable (e.g., a Node.js script or Python script implementing
    the MCP spec for browser control).
    """
    return MCPStdioTool(
        name="browser-automation",
        description="Provides tools for web browsing including navigation, clicking, and extraction.",
        command=sys.executable,
        args=["-m", "my_browser_mcp_server"],  # Hypothetical module
        load_tools=True,
        load_prompts=False,
        allowed_tools=[
            "browser_navigate",
            "browser_navigate_back",
            "browser_snapshot",
            "browser_find",
            "browser_click",
            "browser_hover",
            "browser_type",
            "browser_fill_form",
            "browser_select_option",
            "browser_press_key",
            "browser_take_screenshot",
            "browser_tabs",
            "browser_handle_dialog",
            "browser_evaluate",
            "browser_wait_for",
            "browser_close",
        ],
        tool_name_prefix="web",
    )
