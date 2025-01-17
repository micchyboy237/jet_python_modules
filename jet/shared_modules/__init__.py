# Setup all shared packages
from .global_types import *
from .modules import *
from .globals import inject_globals
from .events import EventSettings

# Ensure globals are injected when the package is imported
inject_globals()


# Trigger event event_pre_start_hook
EventSettings.pre_start_hook()


# Add all imported functions to __all__
__all__ = [
    # Global functions
    'logger',
    'time_it',
    'make_serializable',
    'format_json',
    'prettify_value',
    'class_to_string',
    'get_class_name',
    'save_file',

    # Global variables
    'EventSettings',
]
