"""Factory for creating Browser-Use compatible LLM adapters and Agents using jet infrastructure."""

from __future__ import annotations

from browser_use import Tools
from jet.adapters.browser_use.agent import JetBrowserAgent
from jet.adapters.browser_use.chat import ChatLlamaCpp
from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL


def get_browser_use_llm(
    model: str = LLM_MODEL,
    base_url: str | None = LLM_BASE_URL,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    **kwargs,
) -> ChatLlamaCpp:
    """Get a ChatLlamaCpp instance configured for Browser-Use.

    Args:
        model: The llama.cpp model name to use.
        base_url: The base URL of the llama.cpp server.
        temperature: Sampling temperature. Lower is better for structured tasks.
        max_tokens: Maximum tokens to generate.
        **kwargs: Additional arguments passed to ChatLlamaCpp.

    Returns:
        A ChatLlamaCpp instance implementing the BaseChatModel protocol.
    """
    args = {
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs,
    }
    return ChatLlamaCpp(**args)


def create_agent(
    task: str,
    model_name: str | None = None,
    custom_tools: Tools | None = None,
    system_prompt_extension: str | None = None,
    use_vision: bool = True,
    use_thinking: bool = False,
    max_steps: int = 50,
    **kwargs,
) -> JetBrowserAgent:
    """Factory function to create a pre-configured Jet Browser Agent.

    Args:
        task: The task description for the agent.
        model_name: Specific llama.cpp model to use (overrides env default).
        custom_tools: Pre-built tools registry for custom actions.
        system_prompt_extension: Extra context/instructions for the agent.
        use_vision: Whether to enable vision capabilities (recommended for browser tasks).
        use_thinking: Whether to enable the internal thinking/reasoning block.
                      Defaults to False for local models to improve JSON stability.
        max_steps: Maximum number of steps the agent can take.
        **kwargs: Additional arguments passed to the JetBrowserAgent.

    Returns:
        A configured JetBrowserAgent instance ready to run.
    """
    llm = get_browser_use_llm(model=model_name) if model_name else None

    args = {
        "task": task,
        "llm": llm,
        "tools": custom_tools,
        "extend_system_message": system_prompt_extension,
        "use_vision": use_vision,
        "use_thinking": use_thinking,
        "max_steps": max_steps,
        **kwargs,
    }

    return JetBrowserAgent(**args)
