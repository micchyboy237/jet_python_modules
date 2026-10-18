import json
import logging
from typing import Any, Dict

from mcp import ClientSession
from opentelemetry import trace as otel_trace

from .config import SAFE_OPTIONAL

logger = logging.getLogger("web_surfer_agent")


def allowed_params(tool) -> set:
    """Required args + the few optional args we trust the LLM with."""
    required = set(tool.input_schema.get("required", []))
    return required | SAFE_OPTIONAL.get(tool.name, set())


def build_tool_docs(tools) -> str:
    """Build the tool list for the prompt from the server's own schemas."""
    lines = []
    for t in tools:
        props = t.input_schema.get("properties", {})
        required = set(t.input_schema.get("required", []))
        params = ", ".join(
            f"{name}{'' if name in required else '?'}: {props[name].get('type', 'any')}"
            for name in props
            if name in allowed_params(t)
        )
        lines.append(f"- {t.name}({params}): {t.description}")
    return "\n".join(lines)


def is_tool_error(result, text: str) -> bool:
    """Works across SDK versions (is_error vs isError) and falls back to the text."""
    flag = getattr(result, "is_error", None)
    if flag is None:
        flag = getattr(result, "isError", None)
    return bool(flag) or text.lstrip().startswith("### Error")


async def execute_tool(
    session: ClientSession, tools_by_name: Dict, name: str, args: Dict[str, Any]
) -> tuple[bool, str]:
    """
    Run one MCP browser tool with proper telemetry.
    Captures tool arguments and explicitly sets tool output for observability.
    """
    tool_obj = tools_by_name.get(name)
    if tool_obj is None:
        return False, f"Unknown tool '{name}'. Use one of: {sorted(tools_by_name)}"

    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute("mcp.tool.name", name)
        span.set_attribute("mcp.tool.arguments", json.dumps(args, default=str))

    allowed = allowed_params(tool_obj)
    dropped = set(args) - allowed
    if dropped:
        logger.warning("✂️ Dropping unsupported args %s for %s", sorted(dropped), name)
        args = {k: v for k, v in args.items() if k in allowed}

    missing = [k for k in tool_obj.input_schema.get("required", []) if k not in args]
    if missing:
        return False, f"Missing required args {missing} for {name}."

    logger.info("🛠️ Executing tool: %s with args: %s", name, args)
    try:
        result = await session.call_tool(name, arguments=args)
        text = result.content[0].text if result.content else ""

        if span.is_recording():
            span.set_attribute("tool.output", text[:2000])
            span.set_attribute("tool.success", not is_tool_error(result, text))
            if is_tool_error(result, text):
                span.set_attribute("tool.error_message", text[:500])

        if is_tool_error(result, text):
            logger.warning("❌ Tool %s failed: %s", name, text[:100])
            return False, text

        logger.info(
            "✅ Tool %s succeeded. Output preview: %s...",
            name,
            text[:100].replace("\n", " "),
        )
        return True, text
    except Exception as e:
        if span.is_recording():
            span.set_attribute("tool.success", False)
            span.set_attribute("tool.error_message", str(e)[:500])
        logger.error("⚠️ Tool execution error for %s: %s", name, e)
        raise
