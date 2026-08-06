"""Human-in-the-loop approval strategies for agent tool calls.

Provides a pluggable, extensible approval mechanism using the Strategy pattern.
Each strategy encapsulates the decision logic for whether a tool call should
be allowed to execute.

Usage:
    from jet.libs.llama_cpp.usage.human_in_the_loop import (
        HumanInTheLoop,
        InteractiveApproval,
        CallbackApproval,
        AutoApproval,
    )

    # Interactive terminal prompt
    agent = Agent(client, approval=InteractiveApproval())

    # Custom callback function
    agent = Agent(client, approval=CallbackApproval(my_approval_fn))

    # Always approve (default behavior)
    agent = Agent(client, approval=AutoApproval())
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

logger = logging.getLogger(__name__)


class HumanInTheLoop(ABC):
    """Abstract base for human-in-the-loop approval strategies.

    Subclasses implement the `approve` method to provide different
    approval mechanisms (interactive, callback-based, automatic, etc.).
    """

    @abstractmethod
    def approve(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Decide whether to approve a tool call.

        Args:
            tool_name: Name of the tool being called.
            arguments: Parsed arguments for the tool call.

        Returns:
            True if the tool call should proceed, False to reject.
        """
        ...

    def on_rejected(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Build the error result returned when a tool call is rejected.

        Override this to customize rejection behavior (e.g., provide a
        fallback value instead of an error).

        Args:
            tool_name: Name of the rejected tool.
            arguments: Arguments that were rejected.

        Returns:
            Dict to be used as the tool result message.
        """
        return {"error": f"Tool call '{tool_name}' was rejected."}


class AutoApproval(HumanInTheLoop):
    """Always approve all tool calls without any interaction.

    This is the default strategy — tools execute immediately with no
    human involvement.
    """

    def approve(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        logger.debug(f"🟢 Auto-approved: {tool_name}({json.dumps(arguments)[:80]})")
        return True


class InteractiveApproval(HumanInTheLoop):
    """Prompt the user in the terminal to approve or reject each tool call.

    Uses Python's built-in `input()` to ask for a y/n decision.

    Attributes:
        prompt_template: Customizable prompt shown to the user.
    """

    def __init__(self, prompt_template: str | None = None):
        self.prompt_template = prompt_template or (
            "🛑 Approve tool call '{tool_name}' with arguments {arguments}? (y/n): "
        )

    def approve(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        prompt = self.prompt_template.format(
            tool_name=tool_name,
            arguments=json.dumps(arguments),
        )
        user_input = input(prompt)
        approved = user_input.lower() == "y"
        if approved:
            logger.info(f"🟢 User approved: {tool_name}")
        else:
            logger.warning(f"🔴 User rejected: {tool_name}")
        return approved


class CallbackApproval(HumanInTheLoop):
    """Delegate approval decisions to a user-provided callback function.

    The callback receives (tool_name, arguments) and must return a bool.

    Example:
        def my_approval(tool_name: str, args: dict) -> bool:
            # Block any "delete_file" calls
            if tool_name == "delete_file":
                return False
            # Only allow known locations for weather
            if tool_name == "get_weather":
                return args.get("location", "").lower() in ["tokyo", "london"]
            return True

        agent = Agent(client, approval=CallbackApproval(my_approval))
    """

    def __init__(self, callback: Callable[[str, dict[str, Any]], bool]):
        if not callable(callback):
            raise ValueError("callback must be a callable returning bool")
        self.callback = callback

    def approve(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        try:
            approved = self.callback(tool_name, arguments)
            if not isinstance(approved, bool):
                logger.warning(
                    f"⚠️ Approval callback returned non-bool {type(approved).__name__}, "
                    f"treating as False"
                )
                return False
            if approved:
                logger.debug(f"🟢 Callback approved: {tool_name}")
            else:
                logger.warning(f"🔴 Callback rejected: {tool_name}")
            return approved
        except Exception as exc:
            logger.error(
                f"❌ Approval callback raised exception for {tool_name}: {exc}"
            )
            return False
