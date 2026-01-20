# jet_python_modules/jet/libs/smolagents/jet_examples/search_local_files.py

from smolagents import OpenAIModel, ToolCallingAgent, LogLevel
from jet.libs.smolagents.tools.local_file_search_tool import LocalFileSearchTool

# Reuse the same local LLM configuration style as search_tool_searxng.py
model = OpenAIModel(
    model_id="local-model",
    api_base="http://shawn-pc.local:8080/v1",
    api_key="not-needed",
    temperature=0.3,
    max_tokens=2048,
)

# Instantiate the local file search tool
local_file_search = LocalFileSearchTool()

agent = ToolCallingAgent(
    tools=[local_file_search],
    model=model,
    add_base_tools=False,
    verbosity_level=LogLevel.DEBUG,
)

if __name__ == "__main__":
    # Example 1: Find Python files under a project directory
    result = agent.run(
        """
        Search my local project for Python files.
        Base directory: ./jet_python_modules
        Pattern: **/*.py
        """
    )
    print(result)

    # Example 2: Search for files containing a specific keyword
    result = agent.run(
        """
        Look for any files under ./jet_python_modules
        that mention "EmbeddingCache" in their contents.
        Limit results to 10 files.
        """
    )
    print(result)
