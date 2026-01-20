# jet_python_modules/jet/libs/smolagents/jet_examples/search_local_markdown_python.py

from smolagents import OpenAIModel, ToolCallingAgent, LogLevel
from jet.libs.smolagents.tools.local_file_search_tool import LocalFileSearchTool

# Reuse the same local OpenAI-compatible LLM configuration
model = OpenAIModel(
    model_id="local-model",
    api_base="http://shawn-pc.local:8080/v1",
    api_key="not-needed",
    temperature=0.2,
    max_tokens=2048,
)

local_file_search = LocalFileSearchTool()

agent = ToolCallingAgent(
    tools=[local_file_search],
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
