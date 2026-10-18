from jet.adapters.smolagents.chat import ChatLlamaCpp
from smolagents import CodeAgent, ToolCallingAgent, ToolCollection

# Define the strict allow-list for browser tools
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

# 1. Setup Local Llama.cpp Model
model = ChatLlamaCpp()

# 2. Connect to your running Playwright MCP Server
mcp_config = {"url": "http://localhost:8931/mcp", "transport": "streamable-http"}

print("Connecting to Playwright MCP at http://localhost:8931...")

try:
    # 3. Load Tools
    # Note: Ensure you have run 'pip install smolagents[mcp]' to avoid ImportError
    with ToolCollection.from_mcp(
        mcp_config, trust_remote_code=True, structured_output=False
    ) as tool_collection:
        # Filter tools to only include those in ALLOWED_TOOLS
        filtered_tools = [
            tool for tool in tool_collection.tools if tool.name in ALLOWED_TOOLS
        ]

        print(
            f"Successfully loaded {len(filtered_tools)} allowed tools from MCP server."
        )
        if len(filtered_tools) < len(tool_collection.tools):
            ignored = [
                t.name for t in tool_collection.tools if t.name not in ALLOWED_TOOLS
            ]
            print(f"Ignored {len(ignored)} tools: {ignored}")

        # 4. Create the Web Surfer Agent
        web_surfer = ToolCallingAgent(
            tools=filtered_tools,
            model=model,
            name="web_surfer",
            description="Navigates websites, clicks elements, and extracts content using a real browser via Playwright.",
            max_steps=15,
            instructions="Always take a snapshot after navigating to ensure the page loaded.",
        )

        # 5. Create the Manager Agent (Orchestrator)
        manager = CodeAgent(
            tools=[],
            model=model,
            managed_agents=[web_surfer],
            additional_authorized_imports=["json"],
            instructions="You have a managed agent named 'web_surfer'. To use it, call it like a function: web_surfer(task='Your detailed task here'). Do not try to redefine it.",
        )

        # 6. Execute Task
        print("\nManager agent starting task...")
        result = manager.run(
            "Go to https://example.com and tell me the title of the page."
        )
        print(f"\nResult: {result}")

except ImportError as e:
    print(f"Import Error: {e}")
    print("Please run: pip install 'smolagents[mcp]'")
except Exception as e:
    print(f"An error occurred: {e}")
