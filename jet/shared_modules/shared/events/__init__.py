from jet.logger.config import configure_logger
from .events import EventSettings

# Trigger event event_pre_start_hook
EventSettings.pre_start_hook(configure_logger)


# Add all imported functions to __all__
__all__ = [
    # Global functions
    'EventSettings',
]
