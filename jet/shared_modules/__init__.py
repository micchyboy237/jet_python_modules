# Setup all shared packages
from .modules import *
from .globals import inject_globals


# Ensure globals are injected when the package is imported
inject_globals()

# Add all imported functions to __all__
__all__ = [
    'logger',
    'time_it',
    'make_serializable',
    'format_json',
    'prettify_value',
    'class_to_string',
    'get_class_name',
    'save_file'
]
