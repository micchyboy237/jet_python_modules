import builtins

from shared.setup.events import EventSettings
from shared.time_tracker import TimeTracker

from jet.logger import logger
from jet.utils.print_utils import print_dict_types
from jet.utils.debug_utils import get_non_function_locals
from jet.utils.class_utils import (
    is_class_instance,
    is_dictionary,
    class_to_string,
    validate_class,
    get_class_name,
    validate_iterable_class,
    get_iterable_class_name,
    get_builtin_attributes,
    get_non_empty_attributes,
    get_non_empty_primitive_attributes,
    get_non_empty_object_attributes,
    get_internal_attributes,
    get_callable_attributes,
    get_non_callable_attributes,
)
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
    if not hasattr(builtins, "EventSettings"):
        builtins.EventSettings = EventSettings
    if not hasattr(builtins, "TimeTracker"):
        builtins.TimeTracker = TimeTracker
    if not hasattr(builtins, "logger"):
        builtins.logger = logger
    if not hasattr(builtins, "get_non_function_locals"):
        builtins.get_non_function_locals = get_non_function_locals
    if not hasattr(builtins, "is_class_instance"):
        builtins.is_class_instance = is_class_instance
    if not hasattr(builtins, "is_dictionary"):
        builtins.is_dictionary = is_dictionary
    if not hasattr(builtins, "class_to_string"):
        builtins.class_to_string = class_to_string
    if not hasattr(builtins, "validate_class"):
        builtins.validate_class = validate_class
    if not hasattr(builtins, "get_class_name"):
        builtins.get_class_name = get_class_name
    if not hasattr(builtins, "validate_iterable_class"):
        builtins.validate_iterable_class = validate_iterable_class
    if not hasattr(builtins, "get_iterable_class_name"):
        builtins.get_iterable_class_name = get_iterable_class_name
    if not hasattr(builtins, "get_builtin_attributes"):
        builtins.get_builtin_attributes = get_builtin_attributes
    if not hasattr(builtins, "get_non_empty_attributes"):
        builtins.get_non_empty_attributes = get_non_empty_attributes
    if not hasattr(builtins, "get_non_empty_primitive_attributes"):
        builtins.get_non_empty_primitive_attributes = get_non_empty_primitive_attributes
    if not hasattr(builtins, "get_non_empty_object_attributes"):
        builtins.get_non_empty_object_attributes = get_non_empty_object_attributes
    if not hasattr(builtins, "get_internal_attributes"):
        builtins.get_internal_attributes = get_internal_attributes
    if not hasattr(builtins, "get_callable_attributes"):
        builtins.get_callable_attributes = get_callable_attributes
    if not hasattr(builtins, "get_non_callable_attributes"):
        builtins.get_non_callable_attributes = get_non_callable_attributes
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
