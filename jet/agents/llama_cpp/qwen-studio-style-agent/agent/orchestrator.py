import logging

from tools.registry import ToolRegistry

from agent.config import Config
from agent.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AgenticOrchestrator:
    def __init__(self, registry: ToolRegistry):
        self.llm = LLMClient()
        self.registry = registry
        self.max_iterations = Config.MAX_AGENT_ITERATIONS

    def run(self, user_query: str) -> str:
        messages = [
            {"role": "system", "content": Config.SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
        tools = self.registry.get_schemas()

        logger.info(f"Starting agent loop for: {user_query[:100]}...")

        for iteration in range(1, self.max_iterations + 1):
            logger.debug(f"Iteration {iteration}/{self.max_iterations}")
            msg = self.llm.chat(messages, tools=tools)

            if not msg.tool_calls:
                logger.info(f"Completed after {iteration} iterations")
                return msg.content

            messages.append(msg)

            for tc in msg.tool_calls:
                logger.info(f"Calling tool: {tc.function.name}")
                result = self.registry.execute(tc.function.name, tc.function.arguments)
                logger.debug(f"Tool result ({len(result)} chars): {result[:200]}...")
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        final_msg = self.llm.chat(messages, tools=None)
        return final_msg.content + "\n\n⚠️ *(Max reasoning steps reached.)*"
