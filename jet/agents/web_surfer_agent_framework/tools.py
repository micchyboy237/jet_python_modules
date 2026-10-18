from agent_framework import MCPStdioTool
from jet.adapters.llama_cpp.config import PHOENIX_BASE_URL
from jet_telemetry import initialize_telemetry

initialize_telemetry(service_name="web-surfer-agent-mcp", endpoint=PHOENIX_BASE_URL)


def get_mcp_browser_tool() -> MCPStdioTool:
    """
    Returns an MCPStdioTool configured to connect to the Playwright MCP server.

    Prerequisites:
    1. Node.js installed.
    2. Run: npm install -g @playwright/mcp
    3. Run: npx playwright install --with-deps (if not already done)
    """
    return MCPStdioTool(
        name="browser-automation",
        description="Provides tools for web browsing including navigation, clicking, and extraction using Playwright.",
        command="npx",
        args=["@playwright/mcp"],
        load_tools=True,
        load_prompts=False,
        # We allow all tools by default, or you can restrict them here
        # allowed_tools=None, # Allow all tools
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
