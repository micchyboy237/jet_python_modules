from pydantic import BaseModel
from autogen_core.models import LLMMessage
from autogen_core.memory._base_memory import ChatCompletionContext
from typing import List


class ConcreteChatCompletionContext(ChatCompletionContext):
    """Concrete implementation of ChatCompletionContext."""

    def __init__(self, messages: List[LLMMessage]):
        super().__init__(messages)

    def get_messages(self) -> List[LLMMessage]:
        """Return the list of messages in the context."""
        return self._messages
