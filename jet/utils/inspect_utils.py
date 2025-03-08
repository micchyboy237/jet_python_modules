import inspect
import os
from collections import defaultdict
from typing import Optional, TypedDict

from shared.setup.types import BaseEventData
from jet.logger import logger


class INSPECT_ORIGINAL_SCRIPT_PATH_RESPONSE(TypedDict):
    first: BaseEventData
    last: BaseEventData


def inspect_original_script_path() -> Optional[INSPECT_ORIGINAL_SCRIPT_PATH_RESPONSE]:
    # Get the stack frames
    stack_info = inspect.stack()

    matching_frames = [
        frame for frame in stack_info
        if any(partial_path in frame.filename for partial_path in ["Jet_Projects/", "repo-libs"]) and "site-packages/" not in frame.filename
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
        if any(partial_path in frame.filename for partial_path in ["Jet_Projects/", "repo-libs"]) and "site-packages/" not in frame.filename:
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
        if any(partial_path in frame.filename for partial_path in ["Jet_Projects/", "repo-libs"]) and "site-packages/" not in frame.filename
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
        if any(partial_path in frame.filename for partial_path in ["Jet_Projects/", "repo-libs"]) and "site-packages/" not in frame.filename
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


__all__ = [
    "inspect_original_script_path",
    "print_inspect_original_script_path",
    "print_inspect_original_script_path_grouped",
    "get_stack_frames",
    "find_stack_frames",
    "get_current_running_function",
]

# Example usage
if __name__ == "__main__":
    print("Original script path:", inspect_original_script_path())
    print_inspect_original_script_path_grouped()
