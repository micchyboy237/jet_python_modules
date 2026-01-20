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
    base_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source/en"

    # ───────────────────────────────────────────────────────────────
    # Main example: Find all .md files containing Python code examples
    # ───────────────────────────────────────────────────────────────
    run_example(
        f"Find all markdown files (*.md) under this directory: {base_dir} "
        "that contain Python code examples (look for code blocks starting with ```python"
    )

    # ───────────────────────────────────────────────────────────────
    # Additional useful examples
    # ───────────────────────────────────────────────────────────────

    # Example 2: Just list all markdown files (no content filter)
    run_example(
        f"List all markdown files (*.md) under: {base_dir}"
    )

    # Example 3: Markdown files mentioning "ToolCallingAgent"
    run_example(
        f"Find markdown files under {base_dir} "
        "that contain the text 'ToolCallingAgent' (case insensitive)"
    )

    # Example 4: Markdown files with any code block (python, bash, text, etc.)
    run_example(
        f"Search in {base_dir} for *.md files "
        "that contain any code block (look for lines starting with ```)"
    )