import pytest
import asyncio
from typing import List
from llama_index.core.memory import Memory
from llama_index.core.workflow import Workflow, Context
from jet.libs.llama_index.workflow_context_memory import run_workflow_context_memory


@pytest.fixture
def setup_memory_and_workflow():
    """Fixture to clean up memory and workflow state."""
    memory = Memory.from_defaults(
        session_id="workflow_session", token_limit=40000)

    class SimpleWorkflow(Workflow):
        pass
    workflow = SimpleWorkflow()
    yield memory, workflow
    # Clean up (no database to clear, in-memory SQLite)


class TestWorkflowContextMemory:
    """Tests for combining memory with workflow context."""

    @pytest.mark.asyncio
    async def test_workflow_context_memory(self, setup_memory_and_workflow):
        # Given: A question and a simple workflow
        memory, workflow = setup_memory_and_workflow
        question = "Resume the workflow."
        tools = []
        # Expect a string response (actual content depends on LLM)
        expected_response = str

        # When: Running the workflow context memory example
        result = await run_workflow_context_memory(question, tools, workflow)

        # Then: Verify the response is a string
        assert isinstance(
            result, expected_response), f"Expected {expected_response}, got {type(result)}"
