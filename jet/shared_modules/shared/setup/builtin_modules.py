import builtins

from jet.logger import logger
from jet.utils.print_utils import print_dict_types
from jet.utils.debug_utils import get_non_function_locals
from jet.transformers.object import make_serializable
from jet.utils.commands import copy_to_clipboard, copy_test_result
from jet.transformers.formatters import format_json, format_html
from jet.utils.inspect_utils import (
    inspect_original_script_path,
    print_inspect_original_script_path,
    print_inspect_original_script_path_grouped,
    get_stack_frames,
    find_stack_frames,
    get_current_running_function,
    get_method_info,
)


# Injects global methods/variables only once
def inject_globals():
    if not hasattr(builtins, "logger"):
        builtins.logger = logger
    if not hasattr(builtins, "get_non_function_locals"):
        builtins.get_non_function_locals = get_non_function_locals
    if not hasattr(builtins, "print_dict_types"):
        builtins.print_dict_types = print_dict_types
    if not hasattr(builtins, "make_serializable"):
        builtins.make_serializable = make_serializable
    if not hasattr(builtins, "copy_to_clipboard"):
        builtins.copy_to_clipboard = copy_to_clipboard
    if not hasattr(builtins, "copy_test_result"):
        builtins.copy_test_result = copy_test_result
    if not hasattr(builtins, "format_json"):
        builtins.format_json = format_json
    if not hasattr(builtins, "format_html"):
        builtins.format_html = format_html
    if not hasattr(builtins, "inspect_original_script_path"):
        builtins.inspect_original_script_path = inspect_original_script_path
    if not hasattr(builtins, "print_inspect_original_script_path"):
        builtins.print_inspect_original_script_path = print_inspect_original_script_path
    if not hasattr(builtins, "print_inspect_original_script_path_grouped"):
        builtins.print_inspect_original_script_path_grouped = print_inspect_original_script_path_grouped
    if not hasattr(builtins, "get_stack_frames"):
        builtins.get_stack_frames = get_stack_frames
    if not hasattr(builtins, "find_stack_frames"):
        builtins.find_stack_frames = find_stack_frames
    if not hasattr(builtins, "get_current_running_function"):
        builtins.get_current_running_function = get_current_running_function
    if not hasattr(builtins, "get_method_info"):
        builtins.get_method_info = get_method_info


inject_globals()

__all__ = []
