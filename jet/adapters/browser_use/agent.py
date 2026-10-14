"""Custom Browser-Use Agent implementation using jet's local LLM infrastructure."""

from __future__ import annotations

from typing import Any

from browser_use import Agent as BaseAgent
from browser_use import Tools
from browser_use.llm.base import BaseChatModel
from jet.logger import logger


class JetBrowserAgent(BaseAgent):
    """A Browser-Use agent configured with jet's local LLM and observability.

    This class extends the standard Browser-Use Agent to provide:
    1. Automatic configuration of the local llama.cpp model.
    2. Integration with jet's custom tools registry.
    3. Consistent logging and error handling via jet.
    """

    def __init__(
        self,
        task: str,
        llm: BaseChatModel | None = None,
        tools: Tools | None = None,
        use_vision: bool = True,
        use_thinking: bool = False,
        max_steps: int = 50,
        extend_system_message: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the Jet Browser Agent.

        Args:
            task: The natural language task for the agent to perform.
            llm: The LLM instance to use. If None, defaults to jet's local llama.cpp config.
            tools: A Tools registry containing custom actions.
            use_vision: Whether to enable vision capabilities for the agent.
            use_thinking: Whether to enable the internal thinking/reasoning block.
                          Recommended False for smaller local models to ensure valid JSON.
            max_steps: Maximum number of steps the agent can take.
            extend_system_message: Additional instructions to append to the system prompt.
            **kwargs: Additional arguments passed to the underlying Browser-Use Agent.
        """
        # Store jet-specific attributes before calling super()
        self._jet_llm = llm

        # Initialize tools if not provided
        final_tools = tools or Tools()

        # Prepare kwargs for the parent BaseAgent
        agent_kwargs = {
            "task": task,
            "llm": llm,
            "tools": final_tools,
            "use_vision": use_vision,
            "use_thinking": use_thinking,
            "extend_system_message": extend_system_message,
            "max_actions_per_step": 5,  # Conservative default for local models
            **kwargs,
        }

        # Call the parent constructor
        super().__init__(**agent_kwargs)

        self.max_steps = max_steps
        logger.info(f"🤖 JetBrowserAgent initialized for task: {task[:50]}...")

    async def run(self, max_steps: int | None = None, **kwargs: Any) -> Any:
        """Execute the agent task in the browser with jet logging.

        Args:
            max_steps: Override the default max steps for this run.
            **kwargs: Additional arguments passed to the parent run method.

        Returns:
            The final result of the agent's execution.
        """
        steps = max_steps or self.max_steps
        logger.info(f"🚀 Starting Jet Browser Agent execution (max steps: {steps})")

        try:
            # Use the parent's run method which handles the core agent loop
            history = await super().run(max_steps=steps, **kwargs)

            final_result = history.final_result()
            logger.info(f"✅ Agent completed successfully.")

            return final_result

        except Exception as e:
            logger.error(f"❌ Agent execution failed: {str(e)}")
            raise
