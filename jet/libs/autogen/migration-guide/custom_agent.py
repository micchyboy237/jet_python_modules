"""Define a custom agent with tailored message handling in AutoGen v0.4.

This module shows how to create a `CustomAgent` in AutoGen v0.4 by extending `BaseChatAgent`, replacing the v0.2 `ConversableAgent` with registered reply functions. It implements asynchronous message handling and reset functionality, providing a flexible way to define custom agent behavior.
"""

from typing import Sequence
from autogen_core import CancellationToken
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.base import Response

class CustomAgent(BaseChatAgent):
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        return Response(chat_message=TextMessage(content="Custom reply", source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)