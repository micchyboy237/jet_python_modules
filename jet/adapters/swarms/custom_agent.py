from swarms import Agent
from swarms.structs.agent import AgentInitializationError
from jet.logger import logger
from typing import Any, List, Dict
import json

class CustomAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.llm and hasattr(self.llm, "model_name"):
            logger.info(f"[CustomAgent] Setting model_name to {self.llm.model_name}")
            self.model_name = self.llm.model_name
        else:
            logger.warning(f"[CustomAgent] No llm or llm.model_name found, using model_name={self.model_name}")
            if self.model_name is None:
                self.model_name = "custom-llm"  # Set a fallback to avoid NoneType errors
                logger.info(f"[CustomAgent] Defaulting model_name to 'custom-llm' to avoid validation errors")

    def reliability_check(self):
        """Override to skip LiteLLM checks entirely if custom LLM is provided."""
        if self.llm is not None and hasattr(self.llm, "model_name"):
            logger.info(f"[CustomAgent] Skipping LiteLLM reliability checks for custom LLM: {self.llm.model_name}")
            # Perform basic configuration checks only
            if self.system_prompt is None:
                logger.warning(
                    "[CustomAgent] The system prompt is not set. Please set a system prompt for the agent to improve reliability."
                )
            if self.agent_name is None:
                logger.warning(
                    "[CustomAgent] The agent name is not set. Please set an agent name to improve reliability."
                )
            if self.max_loops is None or self.max_loops == 0:
                raise AgentInitializationError(
                    "Max loops is not provided or is set to 0. Please set max loops to 1 or more."
                )
            if self.max_tokens is None or self.max_tokens <= 0:
                logger.warning(
                    f"[CustomAgent] max_tokens not set or invalid, defaulting to 8192 for custom LLM: {self.llm.model_name}"
                )
                self.max_tokens = 8192
            if self.context_length is None or self.context_length == 0:
                raise AgentInitializationError(
                    "Context length is not provided. Please set a valid context length."
                )
            return
        logger.info(f"[CustomAgent] No custom LLM provided, falling back to default reliability_check with model_name={self.model_name}")
        super().reliability_check()

    def llm_handling(self, additional_args=None):
        """Override to skip LiteLLM initialization if custom LLM is provided."""
        if self.llm is not None:
            logger.info(f"[CustomAgent] Using custom LLM: {self.model_name}")
            return self.llm
        logger.info(f"[CustomAgent] No custom LLM provided, falling back to default llm_handling with model_name={self.model_name}")
        return super().llm_handling(additional_args)

    def check_model_supports_utilities(self, img=None):
        """Override to skip function calling checks if custom LLM supports it, ignoring img parameter."""
        if hasattr(self.llm, "supports_function_calling") and self.llm.supports_function_calling:
            logger.info(f"[CustomAgent] Custom LLM supports function calling: {self.model_name}, img={img}")
            return True
        logger.warning(f"[CustomAgent] Custom LLM does not explicitly support function calling, proceeding with default checks, img={img}")
        return super().check_model_supports_utilities(img=img)

    def temp_llm_instance_for_tool_summary(self):
        from jet.adapters.swarms.mlx_function_caller import MLXFunctionCaller
        return MLXFunctionCaller(
            max_tokens=4000,
            temperature=0.3,
        )

    def execute_tools(self, response: Dict[str, Any] = None, tool_calls: List[Dict[str, Any]] = None, loop_count: int = None) -> List[Any]:
        """Override to execute tools using the custom LLM, handling response, tool_calls, and loop_count."""
        if loop_count is not None:
            logger.info(f"[CustomAgent] Executing tools for loop_count: {loop_count}")
        if response and not tool_calls:
            tool_calls = response.get("tool_calls", [])
            logger.info(f"[CustomAgent] Extracted tool_calls from response: {tool_calls}")
        elif not tool_calls:
            logger.warning("[CustomAgent] No tool_calls provided or extracted from response")
            return []
        if not self.llm:
            logger.error("[CustomAgent] No LLM provided for tool execution")
            return []
        results = []
        for tool_call in tool_calls:
            try:
                if tool_call.get("type") == "function":
                    func_name = tool_call["function"]["name"]
                    args = tool_call["function"].get("arguments", {})
                    logger.info(f"[CustomAgent] Processing tool call: {func_name} with args: {args}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError as e:
                            logger.error(f"[CustomAgent] Failed to parse tool arguments for {func_name}: {e}")
                            results.append(f"Error: Invalid tool arguments for {func_name}")
                            continue
                    tool_func = None
                    for tool in self.tools:
                        if hasattr(tool, "__name__") and tool.__name__ == func_name:
                            tool_func = tool
                            break
                    if not tool_func:
                        logger.error(f"[CustomAgent] Tool {func_name} not found in available tools")
                        results.append(f"Error: Tool {func_name} not found")
                        continue
                    try:
                        result = tool_func(**args)
                        logger.info(f"[CustomAgent] Tool {func_name} executed successfully with result: {result}")
                        results.append(result)
                    except Exception as e:
                        logger.error(f"[CustomAgent] Error executing tool {func_name}: {e}")
                        results.append(f"Error: {str(e)}")
                else:
                    logger.warning(f"[CustomAgent] Unsupported tool call type: {tool_call.get('type')}")
                    results.append(f"Error: Unsupported tool call type: {tool_call.get('type')}")
            except Exception as e:
                logger.error(f"[CustomAgent] General error processing tool call {tool_call}: {e}")
                results.append(f"Error: {str(e)}")
        logger.info(f"[CustomAgent] Tool execution results: {results}")
        return results
