"""Demonstrate a nested chat structure in AutoGen v0.4.

This module illustrates a nested chat in AutoGen v0.4 by creating a `NestedCountingAgent` that uses a `RoundRobinGroupChat` of `CountingAgent` instances. It replaces the v0.2 `register_nested_chats`, allowing hierarchical agent interactions where the outer agent delegates tasks to an inner team.
"""

import asyncio
from typing import Sequence
from autogen_core import CancellationToken
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.base import Response

class CountingAgent(BaseChatAgent):
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        if len(messages) == 0:
            last_number = 0
        else:
            assert isinstance(messages[-1], TextMessage)
            last_number = int(messages[-1].content)
        return Response(chat_message=TextMessage(content=str(last_number + 1), source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

class NestedCountingAgent(BaseChatAgent):
    def __init__(self, name: str, counting_team: RoundRobinGroupChat) -> None:
        super().__init__(name, description="An agent that counts numbers.")
        self._counting_team = counting_team

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        result = await self._counting_team.run(task=messages, cancellation_token=cancellation_token)
        assert isinstance(result.messages[-1], TextMessage)
        return Response(chat_message=result.messages[-1], inner_messages=result.messages[len(messages):-1])

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        await self._counting_team.reset()

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

async def main() -> None:
    counting_agent_1 = CountingAgent("counting_agent_1", description="An agent that counts numbers.")
    counting_agent_2 = CountingAgent("counting_agent_2", description="An agent that counts numbers.")
    counting_team = RoundRobinGroupChat([counting_agent_1, counting_agent_2], max_turns=5)
    nested_counting_agent = NestedCountingAgent("nested_counting_agent", counting_team)
    response = await nested_counting_agent.on_messages([TextMessage(content="1", source="user")], CancellationToken())
    assert response.inner_messages is not None
    for message in response.inner_messages:
        print(message)
    print(response.chat_message)

asyncio.run(main())