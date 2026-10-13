"""
Tests for the CodeAgent agentic workflow.
"""

from unittest.mock import patch

import pytest
from agent import CodeAgent


class TestCodeAgent:
    @pytest.mark.asyncio
    async def test_agent_initialization(self, temp_repo_dir):
        """Test that the agent initializes with tools and map."""
        agent = CodeAgent(repo_path=str(temp_repo_dir))
        agent.load_repository_map()

        assert agent.repo_path is not None
        assert agent.tools_instance is not None
        assert agent.map_content != ""  # Map should be generated

    @pytest.mark.asyncio
    async def test_agent_tool_execution_read_file(self, code_agent):
        """Test that the agent can execute the read_file tool."""

        async def mock_achat(*args, **kwargs):
            tool_registry = kwargs.get("tool_registry", {})

            # Simulate the tool execution and final answer
            if "read_file" in tool_registry:
                # Execute the tool synchronously as it is defined in RepoTools
                tool_result = tool_registry["read_file"]("main.py")
                # Return a mock result that simulates the LLM's final answer after seeing the file
                return type(
                    "obj",
                    (object,),
                    {
                        "content": f"Based on the file, main.py contains: {tool_result[:50]}...",
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 50,
                            "total_tokens": 150,
                        },
                    },
                )
            return type("obj", (object,), {"content": "Error", "usage": {}})

        with patch(
            "jet.adapters.llama_cpp.llm_utils_observed.achat", side_effect=mock_achat
        ):
            result = await code_agent.run("What is in main.py?")

            # Check that the answer contains content from the file
            assert "main.py" in result["answer"]
            assert (
                "calculate_sum" in result["answer"]
                or "file contains" in result["answer"].lower()
            )

    @pytest.mark.asyncio
    async def test_agent_tool_execution_grep(self, code_agent):
        """Test that the agent can execute the grep_search tool."""

        async def mock_achat(*args, **kwargs):
            tool_registry = kwargs.get("tool_registry", {})

            # Simulate the tool execution and final answer
            if "grep_search" in tool_registry:
                # Execute the tool synchronously
                results = tool_registry["grep_search"]("calculate_sum")
                # The LLM might say "I found it in..." or "It is defined in..."
                return type(
                    "obj",
                    (object,),
                    {
                        "content": f"The calculate_sum function is located in {len(results)} files.",
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 50,
                            "total_tokens": 150,
                        },
                    },
                )
            return type("obj", (object,), {"content": "Error", "usage": {}})

        with patch(
            "jet.adapters.llama_cpp.llm_utils_observed.achat", side_effect=mock_achat
        ):
            result = await code_agent.run("Find the calculate_sum function")

            # Assert that the answer mentions the function name and its location/count
            assert "calculate_sum" in result["answer"]
            assert (
                "located" in result["answer"].lower()
                or "found" in result["answer"].lower()
            )

    @pytest.mark.asyncio
    async def test_agent_no_repo_error(self):
        """Test agent behavior when no repo is loaded."""
        agent = CodeAgent(repo_path=None)
        result = await agent.run("Test query")

        assert "Error" in result["answer"]
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_agent_source_extraction(self, code_agent):
        """Test that the agent extracts sources from its final answer."""

        async def mock_achat(*args, **kwargs):
            return type(
                "obj",
                (object,),
                {
                    # Use backticks as LLMs often do for file names
                    "content": "You should look at `main.py` and utils.py for this implementation.",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                },
            )

        with patch(
            "jet.adapters.llama_cpp.llm_utils_observed.achat", side_effect=mock_achat
        ):
            result = await code_agent.run("Test query")

            sources = result["sources"]
            # Ensure sources are extracted even with markdown formatting or plain text
            source_files = [s["file"] for s in sources]

            # The regex in agent.py should handle backticks now, but we check the result list
            assert any("main.py" in f for f in source_files), (
                f"Expected main.py in {source_files}"
            )
            assert any("utils.py" in f for f in source_files), (
                f"Expected utils.py in {source_files}"
            )
