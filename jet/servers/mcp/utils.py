import json
import re
from typing import List
from pydantic import ValidationError
from jet.logger import CustomLogger
from jet.servers.mcp.mcp_classes import ToolRequest


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
