# jet_python_modules/jet/libs/smolagents/mem0_adapter.py
"""
Mem0 integration adapter for smolagents.

Provides persistent, searchable memory for agents by wrapping mem0
and bridging it with smolagents' AgentMemory system.

Usage:
    from jet.libs.smolagents.mem0_adapter import Mem0AgentMemory

    # Create mem0-backed memory
    memory = Mem0AgentMemory(
        mem0_config={
            "llm": {...},
            "embedder": {...},
            "vector_store": {...},
        },
        agent_id="my_agent",
        auto_extract=True,  # Use LLM to extract facts
        auto_store_steps=True,  # Store each step as memory
    )

    # Use with agent via step_callbacks
    agent = CodeAgent(
        tools=[...],
        model=model,
        step_callbacks=[memory.create_step_callback()],
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from mem0 import Memory
from mem0.configs.base import MemoryConfig
from smolagents.memory import (
    ActionStep,
    FinalAnswerStep,
    PlanningStep,
    TaskStep,
)

logger = logging.getLogger(__name__)


@dataclass
class Mem0AgentConfig:
    """Configuration for Mem0 agent memory integration."""

    # mem0 configuration
    mem0_config: dict[str, Any] = field(default_factory=dict)

    # Memory management
    agent_id: str = "default_agent"
    auto_extract: bool = True  # Use LLM to extract facts from steps
    auto_store_steps: bool = True  # Automatically store each step
    store_task: bool = True
    store_planning: bool = True
    store_actions: bool = True
    store_final_answer: bool = True

    # Search defaults
    default_top_k: int = 5
    default_threshold: float = 0.3

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class Mem0AgentMemory:
    """
    Persistent, vector-searchable memory adapter for smolagents using mem0.

    This wraps mem0 and provides:
    - Automatic storage of agent steps as memories
    - LLM-powered fact extraction from conversations
    - Semantic search across all agent memories
    - Per-agent memory isolation via agent_id
    """

    def __init__(
        self,
        mem0_config: dict[str, Any] | None = None,
        agent_id: str = "default_agent",
        auto_extract: bool = True,
        auto_store_steps: bool = True,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ):
        self.config = Mem0AgentConfig(
            mem0_config=mem0_config or {},
            agent_id=agent_id,
            auto_extract=auto_extract,
            auto_store_steps=auto_store_steps,
            metadata=metadata or {},
            **kwargs,
        )

        self._memory: Memory | None = None
        self._step_count = 0
        self._run_id: str | None = None

        logger.info(
            "Mem0AgentMemory initialized | agent_id=%s | auto_extract=%s",
            agent_id,
            auto_extract,
        )

    @property
    def memory(self) -> Memory:
        """Lazy initialization of mem0 Memory instance."""
        if self._memory is None:
            logger.info("Initializing mem0 Memory instance")
            config = MemoryConfig(**self.config.mem0_config)
            self._memory = Memory(config)
        return self._memory

    def set_run_id(self, run_id: str) -> None:
        """Set the current run ID for memory grouping."""
        self._run_id = run_id
        logger.debug("Run ID set: %s", run_id)

    def add_memory(
        self,
        content: str | list[dict],
        infer: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """
        Add a memory with optional LLM extraction.

        Args:
            content: Text or conversation to store
            infer: Whether to use LLM extraction (defaults to config)
            metadata: Additional metadata for this memory

        Returns:
            mem0 add result
        """
        infer = infer if infer is not None else self.config.auto_extract

        combined_metadata = {
            "agent_id": self.config.agent_id,
            "run_id": self._run_id,
            "step_number": self._step_count,
            **self.config.metadata,
            **(metadata or {}),
        }

        logger.debug(
            "Adding memory | infer=%s | content_length=%d",
            infer,
            len(str(content)),
        )

        result = self.memory.add(
            content,
            user_id=self.config.agent_id,
            agent_id=self.config.agent_id,
            run_id=self._run_id,
            metadata=combined_metadata,
            infer=infer,
        )

        logger.info(
            "Memory added | results=%d",
            len(result.get("results", [])),
        )

        return result

    def search(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        **kwargs,
    ) -> dict:
        """
        Search memories semantically.
        Args:
            query: Search query
            top_k: Number of results
            threshold: Minimum similarity threshold
        Returns:
            mem0 search results
        """
        top_k = top_k or self.config.default_top_k
        threshold = threshold or self.config.default_threshold
        logger.debug(
            "Searching memories | query=%s | top_k=%d | threshold=%.2f",
            query[:50],
            top_k,
            threshold,
        )
        result = self.memory.search(
            query,
            filters={"user_id": self.config.agent_id},
            threshold=threshold,
            **kwargs,
        )
        logger.info(
            "Search complete | results=%d",
            len(result.get("results", [])),
        )
        return result

    def get_all(self, **kwargs) -> dict:
        """Get all memories for this agent."""
        return self.memory.get_all(
            filters={"user_id": self.config.agent_id},
            **kwargs,
        )

    def update_memory(self, memory_id: str, data: str) -> dict:
        """Update an existing memory."""
        logger.debug("Updating memory | id=%s", memory_id[:8])
        return self.memory.update(memory_id, data=data)

    def delete_memory(self, memory_id: str) -> dict:
        """Delete a memory."""
        logger.debug("Deleting memory | id=%s", memory_id[:8])
        return self.memory.delete(memory_id)

    def reset(self) -> None:
        """Reset all memories for this agent."""
        logger.info("Resetting all memories for agent: %s", self.config.agent_id)
        self.memory.reset()
        self._step_count = 0

    def close(self) -> None:
        """Close the memory connection."""
        if self._memory is not None:
            self.memory.close()

    # ─── Step Callback Integration ───────────────────────────────────────

    def create_step_callback(self) -> Callable:
        """
        Create a callback compatible with smolagents step_callbacks.

        Returns a function that can be passed to:
        - Agent(step_callbacks=[callback])
        - Agent(step_callbacks={ActionStep: callback})
        """

        def on_step(memory_step, agent=None):
            self._on_agent_step(memory_step, agent)

        return on_step

    def create_step_callbacks_dict(self) -> dict:
        """Create a dict of step callbacks for different step types."""
        return {
            TaskStep: lambda step, agent: self._on_task_step(step, agent),
            PlanningStep: lambda step, agent: self._on_planning_step(step, agent),
            ActionStep: lambda step, agent: self._on_action_step(step, agent),
            FinalAnswerStep: lambda step, agent: self._on_final_answer_step(
                step, agent
            ),
        }

    def _on_agent_step(self, memory_step, agent=None) -> None:
        """Handle any memory step from the agent."""
        if not self.config.auto_store_steps:
            return

        self._step_count += 1

        if isinstance(memory_step, TaskStep):
            self._on_task_step(memory_step, agent)
        elif isinstance(memory_step, PlanningStep):
            self._on_planning_step(memory_step, agent)
        elif isinstance(memory_step, ActionStep):
            self._on_action_step(memory_step, agent)
        elif isinstance(memory_step, FinalAnswerStep):
            self._on_final_answer_step(memory_step, agent)

    def _on_task_step(self, step: TaskStep, agent=None) -> None:
        """Store task as memory."""
        if not self.config.store_task:
            return

        content = f"Task: {step.task}"
        self.add_memory(
            content,
            infer=False,  # Don't extract facts from task
            metadata={"step_type": "task"},
        )

    def _on_planning_step(self, step: PlanningStep, agent=None) -> None:
        """Store planning step as memory."""
        if not self.config.store_planning:
            return

        # Extract key points from plan
        content = f"Plan: {step.plan}"
        self.add_memory(
            content,
            infer=True,  # Extract facts from plan
            metadata={"step_type": "planning"},
        )

    def _on_action_step(self, step: ActionStep, agent=None) -> None:
        """Store action step as memory."""
        if not self.config.store_actions:
            return

        parts = []

        if step.model_output:
            parts.append(f"Thought: {step.model_output[:500]}")

        if step.code_action:
            parts.append(f"Action: {step.code_action[:300]}")

        if step.observations:
            parts.append(f"Observation: {step.observations[:500]}")

        if step.error:
            parts.append(f"Error: {str(step.error)[:300]}")

        if parts:
            content = "\n".join(parts)
            self.add_memory(
                content,
                infer=True,
                metadata={
                    "step_type": "action",
                    "step_number": step.step_number,
                    "has_error": step.error is not None,
                },
            )

    def _on_final_answer_step(self, step: FinalAnswerStep, agent=None) -> None:
        """Store final answer as memory."""
        if not self.config.store_final_answer:
            return

        content = f"Final Answer: {str(step.output)[:1000]}"
        self.add_memory(
            content,
            infer=True,
            metadata={"step_type": "final_answer"},
        )

    # ─── Memory-Enhanced Agent Wrapper ───────────────────────────────────

    def enhance_system_prompt(
        self,
        base_prompt: str,
        task: str,
        top_k: int | None = None,
    ) -> str:
        """
        Enhance the system prompt with relevant memories.

        Args:
            base_prompt: Original system prompt
            task: Current task to find relevant memories for
            top_k: Number of memories to include

        Returns:
            Enhanced system prompt with memory section
        """
        # Search for relevant memories
        results = self.search(task, top_k=top_k or self.config.default_top_k)

        memories = results.get("results", [])
        if not memories:
            return base_prompt

        # Format memories for prompt
        memory_lines = ["\n## Relevant Past Memories\n"]
        for i, mem in enumerate(memories, 1):
            score = mem.get("score", 0)
            text = mem.get("memory", "")
            memory_lines.append(f"{i}. [{score:.2f}] {text}")

        memory_section = "\n".join(memory_lines)

        enhanced = base_prompt + "\n" + memory_section
        logger.debug(
            "Enhanced system prompt | memories_added=%d",
            len(memories),
        )

        return enhanced

    def get_relevant_context(
        self,
        query: str,
        top_k: int | None = None,
        format_as: str = "text",
    ) -> str | list[dict]:
        """
        Get relevant memories as context for the agent.

        Args:
            query: What to search for
            top_k: Number of results
            format_as: 'text' returns formatted string, 'raw' returns dicts

        Returns:
            Formatted context or raw results
        """
        results = self.search(query, top_k=top_k)
        memories = results.get("results", [])

        if format_as == "raw":
            return memories

        lines = ["## Relevant Context from Memory\n"]
        for i, mem in enumerate(memories, 1):
            lines.append(f"- {mem.get('memory', '')}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize configuration for saving."""
        return {
            "class": self.__class__.__name__,
            "config": {
                "mem0_config": self.config.mem0_config,
                "agent_id": self.config.agent_id,
                "auto_extract": self.config.auto_extract,
                "auto_store_steps": self.config.auto_store_steps,
                "metadata": self.config.metadata,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Mem0AgentMemory":
        """Create from serialized configuration."""
        return cls(**data["config"])
