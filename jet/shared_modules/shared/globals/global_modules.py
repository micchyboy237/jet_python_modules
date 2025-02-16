import builtins

from .modules import (
    logger,
    make_serializable,
    format_json,
    class_to_string,
    validate_class,
    get_class_name,
    validate_iterable_class,
    get_iterable_class_name,
    get_non_empty_attributes,
    get_internal_attributes,
    get_callable_attributes,
)
# from jet.utils.class_utils import get_internal_attributes, get_non_empty_attributes, validate_class


# Injects global methods/variables only once
def inject_globals():
    if not hasattr(builtins, "logger"):
        builtins.logger = logger
    if not hasattr(builtins, "make_serializable"):
        builtins.make_serializable = make_serializable
    if not hasattr(builtins, "format_json"):
        builtins.format_json = format_json
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
    if not hasattr(builtins, "get_non_empty_attributes"):
        builtins.get_non_empty_attributes = get_non_empty_attributes
    if not hasattr(builtins, "get_internal_attributes"):
        builtins.get_internal_attributes = get_internal_attributes
    if not hasattr(builtins, "get_callable_attributes"):
        builtins.get_callable_attributes = get_callable_attributes


inject_globals()

__all__ = []
