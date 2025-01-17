import inspect
import os
from collections import defaultdict
from typing import Optional

from global_types import EventData
from jet.logger import logger
from jet.transformers.object import make_serializable


def inspect_original_script_path() -> Optional[EventData]:
    # Get the stack frames
    stack_info = inspect.stack()

    # Filter frames that contain "JetScripts/" or "jet_python_modules/" in the filename
    matching_frames = [
        frame for frame in stack_info if "JetScripts/" in frame.filename or "jet_python_modules/" in frame.filename]
    matching_functions = [
        frame for frame in matching_frames if frame.function != "<module>"]

    if matching_functions:
        # # Get the last matching frame
        # last_matching_frame = matching_frames[-1]
        # last_positions = last_matching_frame.positions
        # last_lineno = last_matching_frame.lineno

        # Get the last function frame
        last_function_frame = matching_functions[-1]
        filename = last_function_frame.filename
        last_code_context = last_function_frame.code_context
        last_function = last_function_frame.function
        last_lineno = last_function_frame.lineno
        # Extract the file name of the script from the last matching frame
        script_path = os.path.abspath(filename)
        return {
            "filepath": script_path,
            "function": last_function,
            "lineno": last_lineno,
            "code_context": last_code_context,
        }
    else:
        return None


def print_inspect_original_script_path():
    stack_info = inspect.stack()
    print("Inspecting stack frames:\n")
    for idx, frame in enumerate(stack_info):
        # Only print frames that have "JetScripts/" or "jet_python_modules/" in the filename
        if "JetScripts/" in frame.filename or "jet_python_modules/" in frame.filename:
            logger.info(f"Frame #{idx}:")
            logger.debug(f"  File: {frame.filename}")
            print(f"  Line Number: {frame.lineno}")
            print(f"  Function Name: {frame.function}")
            print(f"  Code Context: {frame.code_context}")
            print("-" * 50)


def print_inspect_original_script_path_grouped():
    stack_info = inspect.stack()
    file_groups = defaultdict(list)

    # Group stack frames by file name
    for idx, frame in enumerate(stack_info):
        file_groups[frame.filename].append({
            'frame': idx,
            'index': frame.index,
            'line': frame.lineno,
            'function': frame.function,
            'code_context': frame.code_context
        })

    # Pretty-print grouped stack frames
    print("Inspecting stack frames (grouped by file names):\n")
    for file_name, frames in file_groups.items():
        print(f"\nFile: {file_name}")
        for frame in frames:
            logger.info(f"  Frame #{frame['frame']}:")
            print(f"    Index: {frame['index']}")
            print(f"    Line Number: {frame['line']}")
            print(f"    Function Name: {frame['function']}")
            print(f"    Code Context: {frame['code_context']}")
            print("-" * 50)


# Example usage
if __name__ == "__main__":
    print("Original script path:", inspect_original_script_path())
    print_inspect_original_script_path()
