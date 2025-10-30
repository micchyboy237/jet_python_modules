"""
LangChain Agent Example (create_agent + AgentState)
✅ Fixed for LangChain 0.3+ / LangGraph runtime
"""
from typing import List, Optional
from typing import Callable, Awaitable

from langchain.chat_models import BaseChatModel
from jet.transformers.formatters import format_json
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langgraph.prebuilt.tool_node import ToolCallRequest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from jet.logger import logger, CustomLogger
import os
import shutil

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

agent_log_file = f"{OUTPUT_DIR}/agent.log"
agent_logger = CustomLogger("agent", filename=agent_log_file)
logger.orange(f"Agent logs: {agent_log_file}")

model_log_file = f"{OUTPUT_DIR}/model.log"
model_logger = CustomLogger("model", filename=model_log_file)
logger.orange(f"Model logs: {model_log_file}")

tool_log_file = f"{OUTPUT_DIR}/tool.log"
tool_logger = CustomLogger("tool", filename=tool_log_file)
logger.orange(f"Tool logs: {tool_log_file}")

# tool_log_file = f"{OUTPUT_DIR}/tools.log"
# tool_logger = CustomLogger("tools", filename=tool_log_file, level=logging.DEBUG)
# logger.orange(f"Tool logs: {tool_log_file}")

# ─────────────────────────────────────────────────────────────────────────────
#  TOOL-CALL LOGGING MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────
class ToolCallLoggingMiddleware(AgentMiddleware):
    """Logs start/end of every tool call (inputs → result)."""

    def __init__(self) -> None:
        super().__init__()

    def before_agent(self, state, runtime):
        agent_logger.info(
            "[BEFORE AGENT] (State=%s)", format_json(state)
        )

    def after_agent(self, state, runtime):
        agent_logger.teal(
            "[AFTER AGENT] (State=%s)", format_json(state)
        )

    # ── SYNC ─────────────────────────────────────────────────────────────────
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_name = request.tool_call.get("name", "unknown")
        tool_id = request.tool_call.get("id", "unknown")
        args = request.tool_call.get("args", {})

        tool_logger.info(
            "[TOOL START] %s\nid: %s\nargs: %s", tool_name, tool_id, format_json(args)
        )
        result = handler(request)
        tool_logger.teal(
            "[TOOL END] %s\nid: %s\nresult: %s", tool_name, tool_id, format_json(result)
        )
        return result

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ToolMessage | Command:
        model_settings = request.model_settings
        system_prompt = request.system_prompt
        messages = request.messages
        tool_choice = request.tool_choice
        tools = request.tools
        agent_state = request.state

        model_logger.info("[MODEL START]")
        model_logger.log("\nModel Settings: ", format_json(model_settings), colors=["GRAY", "DEBUG"])
        model_logger.log("\nSystem Prompt: ", format_json(system_prompt), colors=["GRAY", "DEBUG"])
        model_logger.log("\nMessages: ", format_json(messages), colors=["GRAY", "DEBUG"])
        model_logger.log("\nTool Choice: ", format_json(tool_choice), colors=["GRAY", "DEBUG"])
        model_logger.log("\nTools: ", format_json(tools), colors=["GRAY", "DEBUG"])
        model_logger.log("\nAgent State: ", format_json(agent_state), colors=["GRAY", "DEBUG"])
        result = handler(request)
        model_logger.teal(
            "[MODEL END] result=%s", result
        )
        return result

    # ── ASYNC ───────────────────────────────────────────────────────────────
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        tool_name = request.tool_call.get("name", "unknown")
        tool_id = request.tool_call.get("id", "unknown")
        args = request.tool_call.get("args", {})

        tool_logger.info(
            "[TOOL START] %s (id=%s) args=%s", tool_name, tool_id, args
        )
        result = await handler(request)
        tool_logger.teal(
            "[TOOL END] %s (id=%s) result=%s", tool_name, tool_id, result
        )
        return result

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ToolMessage | Command:
        model_settings = request.model_settings
        system_prompt = request.system_prompt
        messages = request.messages
        tool_choice = request.tool_choice
        tools = request.tools
        agent_state = request.state

        model_logger.info(
            "[MODEL START]\n%s", format_json({
                "model_settings": model_settings,
                "system_prompt": system_prompt,
                "messages": messages,
                "tool_choice": tool_choice,
                "tools": tools,
                "agent_state": agent_state,
            })
        )
        result = await handler(request)
        model_logger.teal(
            "[MODEL END] result=%s", result
        )
        return result


def build_agent(tools: List[BaseTool], model: str | BaseChatModel = "qwen3-instruct-2507:4b", system_prompt: Optional[str] = None, temperature: float = 0.0):
    """Create a LangChain agent that can perform basic arithmetic."""
    model = ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url="http://shawn-pc.local:8080/v1",
        verbosity="high",
    ) if isinstance(model, str) else model
    
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[ToolCallLoggingMiddleware()],
        debug=True,
    )

    return agent

def estimate_tokens(messages: List[BaseMessage]) -> int:
    """Estimate token count for message list."""
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoder.encode(m.content)) for m in messages) + len(messages) * 4  # +4 per message overhead

def compress_context(
    messages: List[BaseMessage],
    retriever_results: str,
    max_tokens: int = 3500,  # Leave ~600 for output + safety
    llm: Optional[BaseChatModel] = None
) -> str:
    """
    Compress retrieved docs + conversation history into a concise summary
    while preserving accuracy via LLM self-summarization.
    """
    _llm = llm or ChatOpenAI(
        model="qwen3-instruct-2507:4b",
        temperature=0.0,
        base_url="http://shawn-pc.local:8080/v1",
        verbosity="high",
    )

    full_context = f"Retrieved Documents:\n{retriever_results}\n\nConversation So Far:\n"
    for msg in messages[:-1]:  # exclude current user query
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        full_context += f"{role}: {msg.content}\n"

    if estimate_tokens([SystemMessage(content=full_context)]) < max_tokens:
        return full_context

    # If too long: use LLM to summarize prior context losslessly
    summary_prompt = f"""
    Summarize the following research context and conversation **without losing any technical details, definitions, examples, or cited techniques**. 
    Preserve accuracy and specificity. Focus on key concepts, mechanisms, and findings.

    Content to summarize:
    {full_context}

    Concise Summary (preserve all facts):
    """
    summary_msg = _llm.invoke(summary_prompt)
    return summary_msg.content
