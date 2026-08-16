import logging
from typing import Any

from jet.adapters.llama_cpp.llm_utils import chat
from jet.libs.llama_cpp.usage.chat_stream_types import StreamCompletionResult
from tools.registry import ToolRegistry

from agent.config import Config

logger = logging.getLogger(__name__)


class AgenticOrchestrator:
    """
    OUTER orchestrator for the Qwen Studio search→read→reason→answer pattern.

    The INNER tool-call loop (parse tool_calls → execute → append tool result →
    re-call LLM) is handled ENTIRELY by jet.adapters.llama_cpp.llm_utils.chat()
    via its max_tool_rounds parameter and internal while-loop.

    This orchestrator only manages OUTER iterations: deciding whether the agent
    needs another full search→extract→synthesize cycle after the inner loop
    resolves all immediate tool calls.
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.max_outer_iterations = Config.MAX_OUTER_ITERATIONS
        self.inner_tool_rounds = Config.INNER_TOOL_ROUNDS

    def run(self, user_query: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": Config.SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
        tools = self.registry.get_schemas()
        tool_registry = self.registry.as_llm_utils_registry()

        logger.info(f"Starting Qwen-style outer loop for: {user_query[:100]}...")

        for outer_iter in range(1, self.max_outer_iterations + 1):
            logger.info(
                f"═══ Outer iteration {outer_iter}/{self.max_outer_iterations} ═══"
            )

            # Delegate entire inner tool-call loop to llm_utils.chat()
            # It handles: streaming, tool parsing, execution, message appending,
            # observability spans per round, and termination when no more tool calls.
            result: StreamCompletionResult = chat(
                prompt_or_messages=messages,
                model=Config.LLAMA_MODEL,
                tools=tools,
                tool_registry=tool_registry,
                max_tool_rounds=self.inner_tool_rounds,
                temperature=0.0,
                project_name="qwen-studio-agent",
                phoenix_url=Config.PHOENIX_URL,
            )

            # Terminal condition: inner loop ended with a final text response
            if not result.has_tool_calls:
                logger.info(
                    f"✅ Completed after {outer_iter} outer iteration(s), "
                    f"{result.usage.get('total_tokens', '?') if result.usage else '?'} tokens"
                )
                return result.content

            # Inner loop exhausted max_tool_rounds but still has unresolved tool calls.
            # Append the partial result and let the next outer iteration continue.
            messages.append({"role": "assistant", "content": result.content})
            tool_names = [tc.name for tc in result.tool_calls]
            logger.warning(
                f"Inner loop hit max_tool_rounds ({self.inner_tool_rounds}) "
                f"with pending tools: {tool_names}. Continuing outer loop."
            )

        # Max outer iterations reached — force final synthesis without tools
        logger.warning(f"Max outer iterations ({self.max_outer_iterations}) reached")
        final: StreamCompletionResult = chat(
            prompt_or_messages=messages,
            model=Config.LLAMA_MODEL,
            temperature=0.0,
            project_name="qwen-studio-agent",
            phoenix_url=Config.PHOENIX_URL,
        )
        return (
            final.content
            + "\n\n⚠️ *(Max reasoning steps reached. Answer may be incomplete.)*"
        )
