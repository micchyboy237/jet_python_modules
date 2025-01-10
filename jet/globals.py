import builtins
from jet.logger import time_it
from jet.transformers import make_serializable
from jet.transformers.formatters import format_json, prettify_value


# Injects global methods / variables only once
def inject_globals():
    if not hasattr(builtins, "time_it"):
        builtins.time_it = time_it
    if not hasattr(builtins, "make_serializable"):
        builtins.make_serializable = make_serializable
    if not hasattr(builtins, "format_json"):
        builtins.format_json = format_json
    if not hasattr(builtins, "prettify_value"):
        builtins.prettify_value = prettify_value


inject_globals()
