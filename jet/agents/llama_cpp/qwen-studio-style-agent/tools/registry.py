import json
import logging
import time
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Tool registry with session-scoped failure tracking and URL blacklisting.
    Prevents infinite retry loops on permanently failed URLs.
    """

    def __init__(self, blacklist_ttl_seconds: int = 300):
        self._schemas: list[dict[str, Any]] = []
        self._functions: dict[str, Callable[..., Any]] = {}

        # Failure tracking: url -> [{timestamp, error_type, tool_name}]
        self._failure_log: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._blacklist_ttl = blacklist_ttl_seconds

        # Track consecutive identical calls to detect model stuck-in-loop behavior
        self._recent_calls: list[tuple[str, str]] = []  # (tool_name, args_hash)
        self._max_consecutive_identical = 2

    def register(self, name: str, func: Callable[..., Any], schema: dict[str, Any]):
        self._schemas.append(schema)
        self._functions[name] = func
        logger.info(f"Registered tool: {name}")

    def get_schemas(self) -> list[dict[str, Any]]:
        return self._schemas

    def as_llm_utils_registry(self) -> dict[str, Callable[..., Any]]:
        """Return dict format expected by jet.adapters.llama_cpp.llm_utils.chat()."""
        return dict(self._functions)

    def execute(self, name: str, arguments: str) -> str:
        """Execute tool with pre-check blacklist and post-execution failure logging."""
        if name not in self._functions:
            return json.dumps(
                {
                    "error": "unknown_tool",
                    "message": f"Tool '{name}' not found. Available: {list(self._functions.keys())}",
                    "retry_recommended": False,
                }
            )

        try:
            args = json.loads(arguments)
        except json.JSONDecodeError as e:
            return json.dumps(
                {
                    "error": "invalid_arguments",
                    "message": f"JSON parse error: {e}",
                    "retry_recommended": False,
                }
            )

        # === PRE-EXECUTION: Blacklist check for URL-based tools ===
        if name == "web_extractor" and "url" in args:
            url = args["url"]
            blacklist_status = self._check_blacklist(url)
            if blacklist_status["is_blacklisted"]:
                logger.warning(f"URL blacklisted: {url} ({blacklist_status['reason']})")
                return json.dumps(
                    {
                        "error": "url_blacklisted",
                        "url": url,
                        "message": blacklist_status["message"],
                        "retry_recommended": False,
                        "tried_previously": blacklist_status["failure_count"],
                    }
                )

        # === PRE-EXECUTION: Detect stuck-in-loop identical calls ===
        args_hash = f"{name}:{json.dumps(args, sort_keys=True)}"
        if len(self._recent_calls) >= self._max_consecutive_identical:
            recent_identical = sum(
                1
                for call in self._recent_calls[-self._max_consecutive_identical :]
                if call == args_hash
            )
            if recent_identical >= self._max_consecutive_identical:
                logger.error(
                    f"Detected {recent_identical} identical consecutive calls to {name}; forcing fallback"
                )
                return json.dumps(
                    {
                        "error": "repeated_call_blocked",
                        "message": f"Same {name} call repeated {recent_identical} times. Try a different approach or URL.",
                        "retry_recommended": False,
                        "blocked_args": args,
                    }
                )

        self._recent_calls.append(args_hash)
        # Keep only last 10 calls in memory
        if len(self._recent_calls) > 10:
            self._recent_calls = self._recent_calls[-10:]

        # === EXECUTE TOOL ===
        try:
            result = self._functions[name](**args)
        except Exception as e:
            logger.error(f"Tool {name} raised exception: {e}", exc_info=True)
            result = json.dumps(
                {
                    "error": "execution_exception",
                    "error_type": "unknown",
                    "message": f"{type(e).__name__}: {str(e)[:200]}",
                    "retry_recommended": False,
                }
            )

        # === POST-EXECUTION: Log failures for URL-based tools ===
        if name == "web_extractor" and "url" in args:
            self._log_failure_if_needed(args["url"], name, result)

        return result

    def _check_blacklist(self, url: str) -> dict[str, Any]:
        """Check if URL should be blocked based on recent failure history."""
        now = time.time()
        recent_failures = [
            f
            for f in self._failure_log[url]
            if now - f["timestamp"] < self._blacklist_ttl
        ]

        if not recent_failures:
            return {"is_blacklisted": False}

        # Blacklist immediately on permanent failure
        permanent_failures = [
            f for f in recent_failures if f.get("error_type") == "permanent"
        ]
        if permanent_failures:
            return {
                "is_blacklisted": True,
                "reason": "permanent_failure",
                "message": "URL previously returned permanent error (403/404/anti-bot). Try alternative source.",
                "failure_count": len(recent_failures),
            }

        # Blacklist after 2+ any failures within TTL window
        if len(recent_failures) >= 2:
            return {
                "is_blacklisted": True,
                "reason": "repeated_failures",
                "message": f"URL failed {len(recent_failures)} times recently. Try alternative source.",
                "failure_count": len(recent_failures),
            }

        return {"is_blacklisted": False}

    def _log_failure_if_needed(self, url: str, tool_name: str, result: str):
        """Parse tool result and log failures for future blacklist decisions."""
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict) and "error" in parsed:
                self._failure_log[url].append(
                    {
                        "timestamp": time.time(),
                        "error_type": parsed.get("error_type", "unknown"),
                        "tool_name": tool_name,
                        "message": parsed.get("message", "")[:100],
                    }
                )
                logger.debug(f"Logged failure for {url}: {parsed.get('error_type')}")
        except (json.JSONDecodeError, TypeError):
            # Non-JSON result = success; no logging needed
            pass

    def clear_session_failures(self):
        """Reset failure log between independent queries (optional)."""
        self._failure_log.clear()
        self._recent_calls.clear()
        logger.info("Cleared session failure tracking")
