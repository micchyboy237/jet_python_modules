from typing import Optional
from autogen_agentchat.agents import AssistantAgent

from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter
from jet.models.model_types import LLMModelType
from jet.transformers.text import to_snake_case


async def create_mlx_assistant_agent(
    model: LLMModelType,
    name: str,
    description: str,
    conversation_id: Optional[str] = None,
    log_dir: Optional[str] = None,
    **kwargs,
) -> AssistantAgent:
    # Initialize the model client
    model_client = MLXAutogenChatLLMAdapter(
        model=model,
        name=name,
        conversation_id=conversation_id,
        log_dir=f"{log_dir}/{to_snake_case(name)}_chats" if log_dir else None,
        **kwargs,
    )

    # Create agents with specific roles
    agent = AssistantAgent(
        name=name,
        model_client=model_client,
        description=description,
    )

    return agent
