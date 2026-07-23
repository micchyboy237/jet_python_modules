# jet_python_modules/jet/libs/smolagents/jet_examples/search_local_markdown_python.py

from jet.adapters.smolagents.factory import create_llm_model
from jet.libs.smolagents.tools.local_file_read_tool import LocalFileReadTool
from jet.libs.smolagents.tools.local_file_search_tool import LocalFileSearchTool
from smolagents import LogLevel, ToolCallingAgent

model = create_llm_model(
    temperature=0.2,
)

search_tool = LocalFileSearchTool()
read_tool = LocalFileReadTool()

agent = ToolCallingAgent(
    tools=[search_tool, read_tool],
    model=model,
    add_base_tools=False,
    verbosity_level=LogLevel.DEBUG,
)

if __name__ == "__main__":
    base_dir = (
        "/Users/jethroestrada/Desktop/External_Projects/AI/"
        "repo-libs/smolagents/docs/source/en"
    )

    def run_example(query: str):
        """Helper to run a query and print result with clear separation"""
        print(f"\n{'═' * 80}")
        print(f"QUERY: {query}")
        print(f"{'═' * 80}\n")
        result = agent.run(query)
        print(result)
        print(f"{'─' * 80}\n")

    # Search for Markdown files containing Python examples and summarize them
    run_example(
        f"Search for Markdown files (*.md) under this directory: {base_dir}. "
        "For each file found that contains Python code blocks (```python), "
        "read its content and provide a concise summary of what the file covers — "
        "key topics, main Python examples shown, and any important concepts. "
        "Limit to 10 files. Present results as a structured summary with "
        "file name, topics, and a brief description for each."
    )

    # Alternative: focused search with content filtering
    run_example(
        f"Find all Markdown files under {base_dir} that mention 'tool' or 'agent' "
        "in their content. Read up to 5 of the most relevant ones and summarize "
        "the key patterns or API usage they demonstrate."
    )
