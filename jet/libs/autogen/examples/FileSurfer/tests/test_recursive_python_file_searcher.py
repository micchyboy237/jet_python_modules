# jet_python_modules/jet/libs/autogen/tests/autogen-ext/test_recursive_python_file_searcher.py

import asyncio
import os
import aiofiles
import pytest

from autogen_agentchat.messages import TextMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient

from jet.libs.autogen.examples.FileSurfer.recursive_python_file_searcher import RecursivePythonFileSearcher


@pytest.mark.asyncio
async def test_recursive_searcher(tmp_path: str) -> None:
    """Test that RecursivePythonFileSearcher finds Python files with the query."""

    # Create some Python files in nested dirs
    base_dir = tmp_path / "repo"
    os.makedirs(base_dir / "nested", exist_ok=True)

    file1 = base_dir / "auth.py"
    file2 = base_dir / "nested" / "utils.py"
    file3 = base_dir / "nested" / "irrelevant.py"

    async with aiofiles.open(file1, "wt") as f:
        await f.write("def login_user():\n    pass\n")

    async with aiofiles.open(file2, "wt") as f:
        await f.write("# helper function\ndef helper():\n    pass\n")

    async with aiofiles.open(file3, "wt") as f:
        await f.write("print('nothing here')\n")

    # Instantiate searcher with dummy client
    model = "llama3.2"
    searcher = RecursivePythonFileSearcher(
        name="PySearcher",
        model_client=OllamaChatCompletionClient(model=model),
        base_path=str(base_dir),
    )

    # Run search
    result = await searcher.run(task="login_user")

    # Verify results
    assert isinstance(result.messages[1], TextMessage)
    content = result.messages[1].content
    assert "auth.py" in content
    assert "utils.py" not in content
    assert "irrelevant.py" not in content


@pytest.mark.asyncio
async def test_recursive_searcher_no_matches(tmp_path: str) -> None:
    """Test RecursivePythonFileSearcher when there are no matches."""

    base_dir = tmp_path / "repo"
    os.makedirs(base_dir, exist_ok=True)

    file1 = base_dir / "empty.py"
    async with aiofiles.open(file1, "wt") as f:
        await f.write("# nothing useful here\n")

    model = "llama3.2"
    searcher = RecursivePythonFileSearcher(
        name="PySearcher",
        model_client=OllamaChatCompletionClient(model=model),
        base_path=str(base_dir),
    )

    result = await searcher.run(task="nonexistent_keyword")
    assert isinstance(result.messages[1], TextMessage)
    assert "No matches found" in result.messages[1].content


@pytest.mark.asyncio
async def test_recursive_searcher_serialization(tmp_path: str) -> None:
    """Test that RecursivePythonFileSearcher can be serialized and deserialized."""

    model = "llama3.2"
    searcher = RecursivePythonFileSearcher(
        name="PySearcher",
        model_client=OllamaChatCompletionClient(model=model),
        base_path=str(tmp_path),
    )

    serialized = searcher.dump_component()
    deserialized = RecursivePythonFileSearcher.load_component(serialized)

    assert isinstance(deserialized, RecursivePythonFileSearcher)
    assert deserialized._base_path == str(tmp_path)
