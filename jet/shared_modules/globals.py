import builtins

from .modules import (
    logger,
    time_it,
    make_serializable,
    format_json,
    prettify_value,
    class_to_string,
    get_class_name,
    save_file,
    is_iterable_but_not_primitive,
)


# Injects global methods/variables only once
def inject_globals():
    if not hasattr(builtins, "logger"):
        builtins.logger = logger
    if not hasattr(builtins, "time_it"):
        builtins.time_it = time_it
    if not hasattr(builtins, "make_serializable"):
        builtins.make_serializable = make_serializable
    if not hasattr(builtins, "format_json"):
        builtins.format_json = format_json
    if not hasattr(builtins, "prettify_value"):
        builtins.prettify_value = prettify_value
    if not hasattr(builtins, "class_to_string"):
        builtins.class_to_string = class_to_string
    if not hasattr(builtins, "get_class_name"):
        builtins.get_class_name = get_class_name
    if not hasattr(builtins, "save_file"):
        builtins.save_file = save_file
    if not hasattr(builtins, "is_iterable_but_not_primitive"):
        builtins.is_iterable_but_not_primitive = is_iterable_but_not_primitive


inject_globals()
