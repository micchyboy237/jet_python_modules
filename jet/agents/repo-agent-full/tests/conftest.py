"""
Test fixtures and setup for integration tests.
"""

import asyncio
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable telemetry during tests to avoid connection errors
os.environ["DD_TRACE_ENABLED"] = "false"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = ""

from agent import CodeAgent
from chunker import CodeChunker
from config import db
from scanner import RepositoryScanner
from vector_store import VectorStore


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_repo_dir(tmp_path):
    """Create a temporary repository with sample code."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    # main.py
    (repo_dir / "main.py").write_text("""
def calculate_sum(a: int, b: int) -> int:
    '''Calculate sum of two numbers'''
    return a + b

def calculate_product(a: int, b: int) -> int:
    '''Calculate product of two numbers'''
    return a * b
""")

    # utils.py
    (repo_dir / "utils.py").write_text("""
def format_output(value: any) -> str:
    '''Format value for display'''
    return f"Result: {value}"
""")

    return repo_dir


@pytest.fixture
def scanner():
    return RepositoryScanner()


@pytest.fixture
def chunker():
    return CodeChunker()


@pytest.fixture
def vector_store():
    """Provides a clean VectorStore instance for each test."""
    original_path = db.PATH
    test_path = f"./test_db_{os.getpid()}.lancedb"
    db.PATH = test_path

    store = VectorStore()
    store.initialize()

    yield store

    if Path(test_path).exists():
        shutil.rmtree(test_path, ignore_errors=True)
    db.PATH = original_path


@pytest.fixture
def code_agent(temp_repo_dir, vector_store, scanner, chunker):
    """Provides a CodeAgent instance with an indexed repository."""
    # Index the repo first
    files = scanner.scan(str(temp_repo_dir))
    chunks = chunker.chunk_files(files)
    asyncio.run(vector_store.add_chunks(chunks))

    agent = CodeAgent(repo_path=str(temp_repo_dir))
    return agent
