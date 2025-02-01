import json
from typing import Any, Dict, List, Optional, Type
from jet.validation import ValidationResponse
from langchain_community.graphs import MemgraphGraph


def format_error(error: Dict[str, Any]) -> str:
    """Format the error message to include the path to the error."""
    error_path = ".".join([str(p) for p in error.get("loc", [])])
    error_chunks = [error["msg"]]
    if error_path:
        error_chunks.insert(0, error_path)
    return ": ".join(error_chunks)


def validate_query(query: str, *, url: Optional[str] = "bolt://localhost:7687", username: Optional[str] = "", password: Optional[str] = "", graph: Optional[MemgraphGraph] = None) -> ValidationResponse:
    """Executes a cypher query using Memgraph Graph API for validation."""
    graph = graph or MemgraphGraph(url=url, username=username,
                                   password=password, refresh_schema=False)
    try:
        result = graph.query(query)

        return ValidationResponse(is_valid=True, data=result, errors=None)

    except Exception as e:
        return ValidationResponse(is_valid=False, data=None, errors=[str(e)])
