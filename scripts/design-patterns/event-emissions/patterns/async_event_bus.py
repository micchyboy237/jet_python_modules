"""
Summary: Core async event bus implementation using decorator-based subscription (@bus.on).
Provides concurrent handler execution via asyncio.gather with isolated error handling
to prevent one failing handler from breaking others.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Base event class with metadata"""

    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = ""


class EventBus:
    def __init__(self):
        # Map event types (strings or classes) to list of handlers
        self._handlers: Dict[str, List[Callable]] = {}

    def on(self, event_type: str):
        """Decorator to register a handler for an event type"""

        def decorator(func: Callable):
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(func)
            logger.info(
                f"Registered handler '{func.__name__}' for event '{event_type}'"
            )
            return func

        return decorator

    async def publish(self, event: Event):
        """Publish an event to all registered handlers concurrently"""
        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            logger.warning(f"No handlers found for event: {event.event_type}")
            return

        tasks = [self._safe_execute(handler, event) for handler in handlers]
        await asyncio.gather(*tasks)

    async def _safe_execute(self, handler: Callable, event: Event):
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logger.error(f"Handler {handler.__name__} failed: {e}", exc_info=True)
