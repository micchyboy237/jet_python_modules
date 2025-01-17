import inspect
import os
from collections import defaultdict

from jet.logger import logger
from jet.transformers.object import make_serializable


def inspect_original_script_path():
    # Get the frame of the caller (stack trace)
    caller_frame = inspect.stack()[1]
    # Extract the file name of the script from the caller's frame
    script_path = caller_frame.filename
    return os.path.abspath(script_path)


def print_inspect_original_script_path():
    stack_info = inspect.stack()
    stack_info = make_serializable(stack_info)
    print("Inspecting stack frames:\n")
    for idx, frame in enumerate(stack_info):
        print(f"Frame #{idx}:")
        print(f"  Index: {frame.index}")
        print(f"  File: {frame.filename}")
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
