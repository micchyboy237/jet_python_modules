from typing import Any, Dict, List, Literal, Optional, Sequence, Union, AsyncGenerator
from autogen_core import CancellationToken
from autogen_core.models import LLMMessage
from autogen_core.model_context import ChatCompletionContext, UnboundedChatCompletionContext
from autogen_core.models import ChatCompletionClient
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import BaseChatMessage, TextMessage, MultiModalMessage, StopMessage, HandoffMessage, ToolCallSummaryMessage, StructuredMessage
from autogen_agentchat.base import Response
import asyncio
import random
import uuid
from jet.llm.mlx.adapters.agents.autogen_group_chat import GroupChat
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter


class GroupChatManager(BaseChatAgent):
    """Manages a group chat, coordinating agent interactions and task completion."""

    def __init__(
        self,
        groupchat: GroupChat,
        llm_config: Dict[str, Any],
        system_message: str = "You are the manager of a group chat, responsible for coordinating agents and ensuring task completion.",
        name: str = "GroupChatManager",
        description: str = "Manages group chat interactions among multiple agents."
    ) -> None:
        """
        Initialize the GroupChatManager.
        Args:
            groupchat: The GroupChat instance to manage.
            llm_config: Configuration for the LLM client.
            system_message: System message defining the manager's role.
            name: Name of the manager.
            description: Description of the manager.
        """
        super().__init__(name=name, description=description)
        self.groupchat = groupchat
        self.llm_config = llm_config
        self.system_message = system_message
        self._model_client = MLXAutogenChatLLMAdapter(
            model=llm_config["config_list"][0]["model"],
        )
        self._model_context = UnboundedChatCompletionContext()
        self._model_context.add_message(LLMMessage(
            content=system_message, role="system"))

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        """The types of messages that the GroupChatManager produces in the Response.chat_message field."""
        return [
            TextMessage,
            MultiModalMessage,
            StopMessage,
            HandoffMessage,
            ToolCallSummaryMessage,
            StructuredMessage,
        ]

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        """
        Process incoming messages and coordinate the group chat.
        Args:
            messages: Sequence of incoming messages.
            cancellation_token: Token for cancelling operation.
        Returns:
            Response containing the final message and inner messages.
        """
        inner_messages: List[BaseChatMessage] = []
        for msg in messages:
            await self.groupchat.add_message(msg)
            inner_messages.append(msg)
        last_speaker = None
        for round_num in range(self.groupchat.max_round):
            if cancellation_token.is_cancelled():
                break
            next_speaker = await self.groupchat.select_speaker(
                last_speaker=last_speaker,
                model_client=self._model_client,
                cancellation_token=cancellation_token
            )
            response = await next_speaker.on_messages(messages=[msg for msg in messages], cancellation_token=cancellation_token)
            inner_messages.append(response.chat_message)
            await self.groupchat.add_message(response.chat_message)
            if response.chat_message.content == "TERMINATE":
                break
            last_speaker = next_speaker
        return Response(
            chat_message=inner_messages[-1],
            inner_messages=inner_messages
        )

    async def on_messages_stream(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[Union[BaseChatMessage, Response], None]:
        """
        Process messages and stream responses.
        Args:
            messages: Sequence of incoming messages.
            cancellation_token: Token for cancelling operation.
        Yields:
            Messages and final response during processing.
        """
        inner_messages: List[BaseChatMessage] = []
        for msg in messages:
            await self.groupchat.add_message(msg)
            inner_messages.append(msg)
            yield msg
        last_speaker = None
        for round_num in range(self.groupchat.max_round):
            if cancellation_token.is_cancelled():
                break
            next_speaker = await self.groupchat.select_speaker(
                last_speaker=last_speaker,
                model_client=self._model_client,
                cancellation_token=cancellation_token
            )
            async for event in next_speaker.on_messages_stream(messages=[msg for msg in messages], cancellation_token=cancellation_token):
                if isinstance(event, Response):
                    inner_messages.append(event.chat_message)
                    await self.groupchat.add_message(event.chat_message)
                    yield event
                    if event.chat_message.content == "TERMINATE":
                        return
                else:
                    inner_messages.append(event)
                    yield event
            last_speaker = next_speaker
        yield Response(
            chat_message=inner_messages[-1],
            inner_messages=inner_messages
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the manager and group chat."""
        self.groupchat.messages = []
        await self._model_context.clear()
        for agent in self.groupchat.agents:
            await agent.on_reset(cancellation_token)
