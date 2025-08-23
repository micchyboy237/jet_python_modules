from typing import Dict, List, Literal, Optional, Sequence, Union, AsyncGenerator
from autogen_core import CancellationToken
from autogen_core.models import LLMMessage
from autogen_core.model_context import ChatCompletionContext, UnboundedChatCompletionContext
from autogen_core.models import ChatCompletionClient
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_agentchat.base import Response
import asyncio
import random
import uuid


class GroupChat:
    """Manages a multi-agent conversation thread with constrained speaker transitions."""

    def __init__(
        self,
        agents: List[BaseChatAgent],
        allowed_or_disallowed_speaker_transitions: Optional[Dict[BaseChatAgent,
                                                                 List[BaseChatAgent]]] = None,
        speaker_transitions_type: Literal["allowed", "disallowed"] = "allowed",
        messages: List[BaseChatMessage] = [],
        max_round: int = 10,
        send_introductions: bool = False,
        select_speaker_message_template: Optional[str] = None,
        select_speaker_prompt_template: Optional[str] = None,
        speaker_selection_method: Literal["auto",
                                          "round_robin", "random"] = "auto",
    ) -> None:
        """
        Initialize the GroupChat.

        Args:
            agents: List of participating agents.
            allowed_or_disallowed_speaker_transitions: Dictionary of allowed or disallowed transitions.
            speaker_transitions_type: Type of transitions ("allowed" or "disallowed").
            messages: Initial message history.
            max_round: Maximum number of conversation rounds.
            send_introductions: Whether to send agent introductions.
            select_speaker_message_template: Template for speaker selection messages.
            select_speaker_prompt_template: Prompt template for speaker selection.
            speaker_selection_method: Method for selecting the next speaker.
        """
        self.agents = agents
        self.allowed_or_disallowed_speaker_transitions = allowed_or_disallowed_speaker_transitions or {}
        self.speaker_transitions_type = speaker_transitions_type
        self.messages = messages
        self.max_round = max_round
        self.send_introductions = send_introductions
        self.select_speaker_message_template = select_speaker_message_template or "Select the next speaker: {agent_names}"
        self.select_speaker_prompt_template = select_speaker_prompt_template or "Choose the next speaker from: {agent_names}"
        self.speaker_selection_method = speaker_selection_method
        self._model_context = UnboundedChatCompletionContext()

        if send_introductions:
            for agent in agents:
                self.messages.append(TextMessage(
                    content=f"Introduction: I am {agent.name}, {agent.description}",
                    source=agent.name,
                    id=str(uuid.uuid4())
                ))

    async def add_message(self, message: BaseChatMessage) -> None:
        """Add a message to the group chat history."""
        self.messages.append(message)
        await self._model_context.add_message(message.to_model_message())

    async def select_speaker(
        self,
        last_speaker: Optional[BaseChatAgent],
        model_client: ChatCompletionClient,
        cancellation_token: CancellationToken
    ) -> BaseChatAgent:
        """
        Select the next speaker based on the configured method.

        Args:
            last_speaker: The last agent to speak.
            model_client: Client for model inference (used for 'auto' selection).
            cancellation_token: Token for cancelling operation.

        Returns:
            The selected agent.
        """
        valid_speakers = self._get_valid_speakers(last_speaker)

        if not valid_speakers:
            raise ValueError(
                "No valid speakers available based on transition constraints.")

        if self.speaker_selection_method == "round_robin":
            if last_speaker is None:
                return valid_speakers[0]
            last_index = self.agents.index(
                last_speaker) if last_speaker in self.agents else -1
            for i in range(1, len(self.agents)):
                next_index = (last_index + i) % len(self.agents)
                if self.agents[next_index] in valid_speakers:
                    return self.agents[next_index]
            return valid_speakers[0]

        elif self.speaker_selection_method == "random":
            return random.choice(valid_speakers)

        else:  # auto
            agent_names = ", ".join([agent.name for agent in valid_speakers])
            prompt = self.select_speaker_prompt_template.format(
                agent_names=agent_names)
            messages = await self._model_context.get_messages()
            messages.append(TextMessage(
                content=prompt, source="system", id=str(uuid.uuid4())))

            result = await model_client.create(
                messages=[msg.to_model_message() for msg in messages],
                cancellation_token=cancellation_token
            )

            selected_name = result.content.strip() if isinstance(result.content, str) else ""
            for agent in valid_speakers:
                if agent.name == selected_name:
                    return agent
            return valid_speakers[0]  # Fallback to first valid speaker

    def _get_valid_speakers(self, last_speaker: Optional[BaseChatAgent]) -> List[BaseChatAgent]:
        """Return valid speakers based on transition constraints."""
        if not self.allowed_or_disallowed_speaker_transitions or last_speaker is None:
            return self.agents

        transitions = self.allowed_or_disallowed_speaker_transitions.get(
            last_speaker, [])
        if self.speaker_transitions_type == "allowed":
            return [agent for agent in transitions if agent in self.agents]
        else:  # disallowed
            return [agent for agent in self.agents if agent not in transitions]

    async def get_messages(self) -> List[BaseChatMessage]:
        """Return the current message history."""
        return self.messages
