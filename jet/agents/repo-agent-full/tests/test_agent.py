"""
Tests for the CodeAgent agentic workflow.
"""

from unittest.mock import patch

import pytest
from agent import CodeAgent


class TestCodeAgent:
    @pytest.mark.asyncio
    async def test_agent_initialization(self, temp_repo_dir):
        """Test that the agent initializes with tools."""
        agent = CodeAgent(repo_path=str(temp_repo_dir))
        assert agent.repo_path is not None
        assert agent.tools_instance is not None

    @pytest.mark.asyncio
    async def test_agent_tool_execution_read_file(self, code_agent):
        """Test that the agent can execute the read_file tool."""
        # We mock the LLM to force it to call the read_file tool
        # In a real integration test, we'd let the LLM decide, but for stability:

        # Mock the achat function to simulate an LLM deciding to use a tool
        async def mock_achat(*args, **kwargs):
            # Simulate the LLM calling 'read_file'
            tool_registry = kwargs.get("tool_registry", {})
            if "read_file" in tool_registry:
                result = tool_registry["read_file"]("main.py")
                # Simulate a second turn where the LLM gives the final answer
                return type(
                    "obj",
                    (object,),
                    {"content": f"The file contains: {result[:50]}..."},
                )
            return type("obj", (object,), {"content": "Error"})

        with patch(
            "jet.adapters.llama_cpp.llm_utils_observed.achat", side_effect=mock_achat
        ):
            result = await code_agent.run("What is in main.py?")

            assert "The file contains:" in result["answer"]
            assert "main.py" in result["answer"]

    @pytest.mark.asyncio
    async def test_agent_tool_execution_grep(self, code_agent):
        """Test that the agent can execute the grep_search tool."""

        async def mock_achat(*args, **kwargs):
            tool_registry = kwargs.get("tool_registry", {})
            if "grep_search" in tool_registry:
                results = tool_registry["grep_search"]("calculate_sum")
                return type(
                    "obj",
                    (object,),
                    {"content": f"Found {len(results)} matches for calculate_sum"},
                )
            return type("obj", (object,), {"content": "Error"})

        with patch(
            "jet.adapters.llama_cpp.llm_utils_observed.achat", side_effect=mock_achat
        ):
            result = await code_agent.run("Find the calculate_sum function")

            assert "Found" in result["answer"]
            assert "calculate_sum" in result["answer"]

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

        # Mock achat to return a specific answer with file references
        async def mock_achat(*args, **kwargs):
            return type(
                "obj",
                (object,),
                {"content": "You should look at main.py and utils.py for this."},
            )

        with patch(
            "jet.adapters.llama_cpp.llm_utils_observed.achat", side_effect=mock_achat
        ):
            result = await code_agent.run("Test query")

            sources = result["sources"]
            assert any("main.py" in s["file"] for s in sources)
            assert any("utils.py" in s["file"] for s in sources)
