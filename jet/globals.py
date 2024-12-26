import builtins
from jet.logger import time_it
from jet.transformers import make_serializable


# Injects global methods / variables only once
def inject_globals():
    if not hasattr(builtins, "time_it"):
        builtins.time_it = time_it
    if not hasattr(builtins, "make_serializable"):
        builtins.make_serializable = make_serializable


inject_globals()
