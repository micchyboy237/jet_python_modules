import builtins
from jet.logger import time_it


# Injects global methods / variables only once
def inject_globals():
    if not hasattr(builtins, "time_it"):
        builtins.time_it = time_it


inject_globals()
