import sys
import traceback
import inspect
import os
import re

from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Optional, TypedDict, get_type_hints

from jet.transformers.object import make_serializable
from shared.setup.types import BaseEventData
from jet.logger import logger


class INSPECT_ORIGINAL_SCRIPT_PATH_RESPONSE(TypedDict):
    first: BaseEventData
    last: BaseEventData


# Paths to include and exclude in logs
INCLUDE_PATHS = ["jet_python_modules/", "Jet_Projects/", "repo-libs/"]
EXCLUDE_PATHS = ["site-packages/"]

MAX_LOG_LENGTH = 100  # Max length for logged values


def validate_filepath(file_path: str) -> bool:
    # # Check if path should be included
    # if not any(path in file_path for path in INCLUDE_PATHS):
    #     return False  # Skip if not in allowed paths

    # Check if path should be excluded
    if any(path in file_path for path in EXCLUDE_PATHS):
        return False  # Skip if in excluded paths

    return True


def truncate_value(value):
    """Truncates long strings and collections for logging."""
    serialized = make_serializable(value)
    max_list_items = 2

    if isinstance(serialized, str):
        return serialized[:MAX_LOG_LENGTH] + "..." if len(serialized) > MAX_LOG_LENGTH else serialized
    elif isinstance(serialized, list):
        return [truncate_value(item) for item in serialized[:max_list_items]] + (["..."] if len(serialized) > max_list_items else [])
    elif isinstance(serialized, dict):
        # Log first 5 key-value pairs
        return {k: truncate_value(v) for k, v in list(serialized.items())[:5]}
    return serialized  # Return unchanged for numbers, bools, etc.


def log_filtered_stack_trace(exc: Exception):
    """Logs only relevant stack trace frames based on include/exclude filters, with truncated function argument values."""

    tb = traceback.extract_tb(exc.__traceback__)
    stack = inspect.trace()  # Get full stack for local variables
    error_message = str(exc)  # Extract error message

    for i, frame in enumerate(tb):
        filename = frame.filename

        if not validate_filepath(filename):
            continue

        line_number = frame.lineno
        function_name = frame.name
        code_context = frame.line  # Code that triggered the error

        logger.newline()
        logger.warning(
            f"Stack [{i}]: File \"{filename}\", line {line_number}, in {function_name}"
        )
        logger.warning("Code:")  # Include line number
        logger.error(code_context)
        logger.warning("Error Message:")
        logger.error(error_message)  # Log error message

        # # Find matching function frame to extract argument names + values
        # for stack_frame in stack:
        #     if stack_frame.function == function_name and stack_frame.filename == filename:
        #         # Extract function args (name + values)
        #         local_vars = stack_frame.frame.f_locals
        #         truncated_args = {k: truncate_value(
        #             v) for k, v in local_vars.items()}  # Truncate long values

        #         logger.newline()
        #         logger.warning("Args:")
        #         logger.pretty(truncated_args)

        #         break  # Stop after finding the matching function


def inspect_original_script_path() -> Optional[INSPECT_ORIGINAL_SCRIPT_PATH_RESPONSE]:
    # Get the stack frames
    stack_info = inspect.stack()

    matching_frames = [
        frame for frame in stack_info
        if validate_filepath(frame.filename)
    ]
    matching_functions = [
        frame for frame in matching_frames
        if not frame.function.startswith('_') and
        frame.function != "<module>" and
        os.path.basename(__file__) not in frame.function
    ]

    if matching_functions:
        # # Get the orig matching frame
        first_matching_frame = matching_frames[-1]
        first_filename = first_matching_frame.filename
        first_code_context = first_matching_frame.code_context
        first_function = first_matching_frame.function
        first_lineno = first_matching_frame.lineno

        # Get the last function frame
        last_function_frame = matching_functions[-1]
        last_filename = last_function_frame.filename
        last_code_context = last_function_frame.code_context
        last_function = last_function_frame.function
        last_lineno = last_function_frame.lineno
        return {
            "first": {
                "filepath":  os.path.abspath(first_filename),
                "filename":  os.path.basename(first_filename),
                "function": first_function,
                "lineno": first_lineno,
                "code_context": first_code_context,
            },
            "last": {
                "filepath":  os.path.abspath(last_filename),
                "filename":  os.path.basename(last_filename),
                "function": last_function,
                "lineno": last_lineno,
                "code_context": last_code_context,
            }
        }
    else:
        return None


def print_inspect_original_script_path():
    stack_info = inspect.stack()
    print("Inspecting stack frames:\n")
    for idx, frame in enumerate(stack_info):
        if validate_filepath(frame.filename):
            logger.info(f"Frame #{idx}:")
            logger.log("  File:", frame.filename,
                       colors=["WHITE", "DEBUG"])
            logger.log("  Function Name:", frame.function,
                       colors=["GRAY", "DEBUG"])
            logger.log("  Line Number:", frame.lineno,
                       colors=["GRAY", "DEBUG"])
            logger.log("  Code Context:", frame.code_context,
                       colors=["GRAY", "DEBUG"])
            print("-" * 50)


def print_inspect_original_script_path_grouped():
    stack_info = inspect.stack()
    file_groups = defaultdict(list)

    # Group stack frames by file name
    for idx, frame in enumerate(stack_info):
        file_groups[frame.filename].append({
            'frame': idx,
            'index': frame.index,
            'filename': frame.filename,
            'lineno': frame.lineno,
            'function': frame.function,
            'code_context': frame.code_context
        })

    # Pretty-print grouped stack frames
    print("Inspecting stack frames (grouped by file names):\n")
    for file_name, frames in file_groups.items():
        print(f"\nFile: {file_name}")
        for frame in frames:
            logger.info(f"Frame #{idx}:")
            logger.log("  File:", frame["filename"],
                       colors=["WHITE", "DEBUG"])
            logger.log("  Function Name:", frame["function"],
                       colors=["GRAY", "DEBUG"])
            logger.log("  Line Number:", frame["lineno"],
                       colors=["GRAY", "DEBUG"])
            logger.log("  Code Context:", frame["code_context"],
                       colors=["GRAY", "DEBUG"])
            print("-" * 50)


def get_stack_frames(max_frames: Optional[int] = None):
    stack_info = inspect.stack()
    frames = [
        frame for frame in stack_info
        if validate_filepath(frame.filename)
    ]
    frames = frames[-max_frames:] if max_frames else frames

    stack_frames = [{
        'index': frame.index,
        'filename': frame.filename,
        'lineno': frame.lineno,
        'function': frame.function,
        'code_context': frame.code_context
    } for frame in frames]

    return stack_frames


def find_stack_frames(text: str):
    stack_info = inspect.stack()
    frames = [
        frame for frame in stack_info
        if validate_filepath(frame.filename)
    ]
    frames = [
        frame for frame in frames for code in frame.code_context if text in code]

    stack_frames = [{
        'index': frame.index,
        'filename': frame.filename,
        'lineno': frame.lineno,
        'function': frame.function,
        'code_context': frame.code_context
    } for frame in frames]

    return stack_frames


def get_current_running_function():
    stack = inspect.stack()
    if len(stack) > 1:  # 1 is for the current function, so anything beyond that is a caller
        current_function = stack[1].function
        print(f"Currently running function: {current_function}")
        return current_function


def get_entry_file_name(remove_extension: bool = False) -> str:
    """
    Returns the file name of the entry point script, optionally without the file extension.
    Args:
        remove_extension (bool): If True, returns the file name without the extension; otherwise, includes it.
    Returns:
        str: The file name of the entry point script, or "server" on error.
    """
    try:
        file_path = Path(sys.modules["__main__"].__file__)
        if remove_extension:
            return file_path.with_suffix("").name
        return file_path.name
    except (KeyError, AttributeError):
        return "server"


def get_entry_file_path(remove_extension: bool = False) -> Optional[str]:
    """
    Returns the absolute file path of the entry point script, optionally without the file extension.
    Args:
        remove_extension (bool): If True, returns the file path without the extension; otherwise, includes it.
    Returns:
        Optional[str]: The absolute file path of the entry point script, or None if invalid.
    """
    try:
        file_path = Path(sys.modules["__main__"].__file__).resolve()
        if validate_filepath(str(file_path)):
            if remove_extension:
                return str(file_path.with_suffix(""))
            return str(file_path)
        return None
    except (KeyError, AttributeError):
        return None


def get_entry_file_dir() -> Optional[str]:
    """
    Returns the absolute directory path of the entry point script.
    Returns None if the entry point cannot be determined or is not a valid path.

    Returns:
        Optional[str]: The absolute directory path of the entry point script, or None if invalid.
    """
    try:
        file_path = Path(sys.modules["__main__"].__file__).resolve()
        dir_path = file_path.parent
        if validate_filepath(str(file_path)):
            return str(dir_path)
        return None
    except (KeyError, AttributeError):
        return None


class ParameterInfo(TypedDict):
    type: str
    description: str


class Parameters(TypedDict):
    type: str
    required: list[str]
    properties: Dict[str, ParameterInfo]


class MethodInfo(TypedDict):
    name: str
    description: str
    parameters: Parameters


def get_method_info(method: Any) -> MethodInfo:
    """Extract information from a typed class method or tool-wrapped callable."""
    # Handle CallableWithToolSpec wrappers
    if hasattr(method, "tool_spec"):
        spec = method.tool_spec["function"]
        name = spec["name"]
        description = spec.get("description", "No description available")
        parameters = spec.get("parameters", {"type": "object", "properties": {}, "required": []})
        return {
            "name": name,
            "description": description,
            "parameters": parameters,
        }

    # Existing logic for regular methods...
    type_mapping = {
        int: "integer",
        str: "string",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        Any: "any"
    }

    type_hints = get_type_hints(method)
    params = inspect.signature(method).parameters
    required_params = [name for name in params if name != "self" and params[name].default is inspect.Parameter.empty]
    properties: Dict[str, ParameterInfo] = {}

    docstring = inspect.getdoc(method) or "No description available"
    param_descriptions = {}
    if docstring:
        doc_lines = docstring.splitlines()
        in_args_section = False
        for line in doc_lines:
            line = line.strip()
            if line.startswith("Args:"):
                in_args_section = True
                continue
            if in_args_section and (line.startswith(("Returns:", "Raises:", "Examples:")) or not line):
                in_args_section = False
                continue
            match = re.match(r"^\s*(\w+)\s*(?:\([^)]*\))?:\s*(.+)$", line)
            if match:
                param_name, param_desc = match.groups()
                param_descriptions[param_name] = param_desc.strip()

    for name in required_params:
        param_type = type_hints.get(name, Any)
        type_name = type_mapping.get(param_type, "any")
        description = param_descriptions.get(name, f"The {name} parameter")
        properties[name] = {"type": type_name, "description": description}

    first_line = docstring.split("\n")[0] if docstring else "No description available"

    return {
        "name": method.__name__,
        "description": first_line,
        "parameters": {
            "type": "object",
            "required": required_params,
            "properties": properties
        }
    }


__all__ = [
    "inspect_original_script_path",
    "print_inspect_original_script_path",
    "print_inspect_original_script_path_grouped",
    "get_stack_frames",
    "find_stack_frames",
    "get_current_running_function",
    "get_entry_file_name",
    "get_entry_file_path",
    "get_entry_file_dir",
    "get_method_info",
]

# Example usage
if __name__ == "__main__":
    print("Original script path:", inspect_original_script_path())
    print_inspect_original_script_path_grouped()
