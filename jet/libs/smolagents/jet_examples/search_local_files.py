# jet_python_modules/jet/libs/smolagents/jet_examples/search_local_files.py
from smolagents import OpenAIModel, ToolCallingAgent, LogLevel
from jet.libs.smolagents.tools.local_file_search_tool import LocalFileSearchTool

# ────────────────────────────────────────────────────────────────────────────────
# Same local LLM configuration as in search_tool_searxng.py
# ────────────────────────────────────────────────────────────────────────────────
model = OpenAIModel(
    model_id="local-model",
    api_base="http://shawn-pc.local:8080/v1",
    api_key="not-needed",
    temperature=0.7,
    max_tokens=2048,
)

# ────────────────────────────────────────────────────────────────────────────────
# Create the agent with only the local file search tool
# ────────────────────────────────────────────────────────────────────────────────
agent = ToolCallingAgent(
    tools=[LocalFileSearchTool()],
    model=model,
    add_base_tools=False,
    verbosity_level=LogLevel.DEBUG,
)


def run_example(query: str):
    """Helper to run a query and print result with clear separation"""
    print(f"\n{'═' * 80}")
    print(f"QUERY: {query}")
    print(f"{'═' * 80}\n")
    result = agent.run(query)
    print(result)
    print(f"{'─' * 80}\n")


if __name__ == "__main__":
    # Example 1: Find all Python files in your home directory
    run_example(
        "Find all Python files (*.py) in my home directory. "
        "Use ~/ as base directory."
    )

    # Example 2: Find all markdown files containing the word "agent"
    run_example(
        "Search in ~/projects for all markdown files (*.md) "
        "that contain the word 'agent' (case insensitive)"
    )

    # Example 3: Find configuration files with specific content
    run_example(
        "Look in ~/ for all files named *.toml or *.yaml or *.yml "
        "that contain the substring 'temperature'"
    )

    # Example 4: Non-existent directory (error handling)
    run_example(
        "Search for *.txt files in /this/path/does/not/exist"
    )

    # Example 5: Very broad search with limit
    run_example(
        "Find the first 10 files (any type) in ~/Downloads "
        "whose names contain 'invoice'"
    )