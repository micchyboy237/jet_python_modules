import json
import re
import sys
from pathlib import Path
from typing import List
from pydantic import ValidationError
from jet.logger import CustomLogger
from .mcp_classes import ToolRequest


def setup_module_path() -> None:
    """
    Add the jet_python_modules directory to sys.path for module resolution.

    Traverses up from the caller's directory to find 'jet_python_modules' and adds it to sys.path.

    Raises:
        ImportError: If the jet_python_modules directory cannot be found.
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir
    while project_root.name != 'jet_python_modules' and project_root.parent != project_root:
        project_root = project_root.parent
    if project_root.name == 'jet_python_modules':
        sys.path.insert(0, str(project_root))
    else:
        raise ImportError("Could not find jet_python_modules directory")


def parse_tool_requests(llm_response: str, logger: CustomLogger) -> List[ToolRequest]:
    """
    Parse multiple JSON tool requests from the LLM response.

    Args:
        llm_response: Raw response string from the LLM.
        logger: Logger instance for debugging and error logging.

    Returns:
        List of valid ToolRequest objects.
    """
    json_matches = re.findall(r'\{(?:[^{}]|\{[^{}]*\})*\}', llm_response)
    # logger.debug(f"Found {len(json_matches)} JSON objects in response")
    tool_requests = []
    for json_str in json_matches:
        try:
            tool_request = ToolRequest.model_validate_json(json_str)
            # logger.debug(f"Valid tool request: {tool_request}")
            tool_requests.append(tool_request)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(
                f"Invalid JSON object skipped: {json_str}, error: {str(e)}")
    return tool_requests
