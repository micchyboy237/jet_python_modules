from typing import List, Optional, Callable, Dict, Any
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from llama_index.core.utilities.token_counting import TokenCounter


class OllamaTokenCounter(TokenCounter):
    """Custom TokenCounter class with overridden methods and additional functionality."""

    def __init__(self, tokenizer: Optional[Callable[[str], list]] = None) -> None:
        """Initialize with optional custom tokenizer."""
        super().__init__(tokenizer=tokenizer)

    def get_string_tokens(self, string: str) -> int:
        """Override to provide custom token counting logic for a string."""
        print(f"Counting tokens for string: {string}")
        # For demonstration, return the length of the string as a proxy for token count
        return len(self.tokenizer(string))

    def estimate_tokens_in_messages(self, messages: List[ChatMessage]) -> int:
        """Override to customize token estimation in a list of messages."""
        print("Estimating tokens in messages...")
        tokens = 0
        for message in messages:
            print(f"Message: {message.role} - {message.content}")
            tokens += self.get_string_tokens(message.content)
            if message.role:
                tokens += self.get_string_tokens(message.role)

            additional_kwargs = {**message.additional_kwargs}

            # Mock function call and tool call token counting
            if "function_call" in additional_kwargs:
                tokens += 5  # Mock additional tokens for function call
            if "tool_calls" in additional_kwargs:
                tokens += 10  # Mock additional tokens for tool call

            tokens += 3  # Add three per message for basic message overhead

            if message.role == MessageRole.FUNCTION or message.role == MessageRole.TOOL:
                tokens -= 2  # Subtract 2 if role is "function" or "tool"

        return tokens

    def estimate_tokens_in_tools(self, tools: List[Dict[str, Any]]) -> int:
        """Override to provide custom tool token counting."""
        print("Estimating tokens in tools...")
        tokens = 0
        for tool in tools:
            print(f"Tool: {tool['name']} - Arguments: {tool['arguments']}")
            tokens += self.get_string_tokens(tool['name'])
            tokens += self.get_string_tokens(str(tool['arguments']))

            tokens += 3  # Mock additional tokens for tool calls
        return tokens

    def main(self, messages: List[ChatMessage], tools: Optional[List[dict]] = None) -> int:
        """
        Main function to demonstrate token counting in messages and tools.

        Args:
            messages (List[ChatMessage]): A list of chat messages.
            tools (Optional[List[dict]]): A list of tools to estimate token count.

        Returns:
            int: The total estimated token count.
        """
        try:
            # Estimate the tokens in messages
            message_tokens = self.estimate_tokens_in_messages(messages)
            print(f"Estimated tokens in messages: {message_tokens}")

            # Estimate the tokens in tools
            tool_tokens = self.estimate_tokens_in_tools(tools)
            print(f"Estimated tokens in tools: {tool_tokens}")

            # Return the total token count
            total_tokens = message_tokens + tool_tokens
            print(f"Total estimated tokens: {total_tokens}")
            return total_tokens

        except Exception as e:
            print(f"Error in token counting: {e}")
            return 0


# Example usage:

if __name__ == "__main__":
    # Sample messages
    messages = [
        ChatMessage(role=MessageRole.USER,
                    content="What is the weather today?"),
        ChatMessage(role=MessageRole.SYSTEM,
                    content="The weather is sunny with a chance of rain.")
    ]

    # Sample tools (mock tools for demonstration)
    tools = [
        {"name": "weather_tool", "arguments": {"location": "New York"}},
        {"name": "time_tool", "arguments": {"timezone": "EST"}}
    ]

    # Instantiate the custom token counter
    custom_token_counter = OllamaTokenCounter()

    # Call the main function to estimate token counts
    total_tokens = custom_token_counter.main(messages, tools)

    # Output the total estimated tokens
    print(f"Total tokens: {total_tokens}")
