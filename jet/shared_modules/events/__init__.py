from .events import EventSettings

# Trigger event event_pre_start_hook
EventSettings.pre_start_hook()


# Add all imported functions to __all__
__all__ = [
    # Global functions
    'EventSettings',
]
