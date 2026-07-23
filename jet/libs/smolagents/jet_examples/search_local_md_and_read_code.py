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

    result = agent.run(
        f"""
        Find all Markdown files under the following directory
        that contain Python code examples.

        Base directory: {base_dir}
        File pattern: **/*.md
        The files must contain Python code blocks
        (look for ```python in the file content).
        Limit results to 50 files.
        """
    )

    print(result)
