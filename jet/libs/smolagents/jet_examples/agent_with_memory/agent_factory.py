from callbacks import auto_extract_simple_facts, auto_save_shared_state
from jet.adapters.smolagents.factory import create_llm_model
from smolagents import CodeAgent
from tools.memory_tools import (
    LongTermRecallTool,
    LongTermSaveTool,
    SharedStateReadTool,
    SharedStateUpdateTool,
)


def create_memory_enabled_agent(
    model=None, extra_tools=None, max_steps: int = 40, verbosity: int = 1
) -> CodeAgent:
    model = create_llm_model()
    tools = [
        LongTermSaveTool(),
        LongTermRecallTool(),
        SharedStateUpdateTool(),
        SharedStateReadTool(),
    ]
    if extra_tools:
        tools.extend(extra_tools)

    return CodeAgent(
        tools=tools,
        model=model,
        step_callbacks=[
            auto_save_shared_state,
            auto_extract_simple_facts,
            # add more callbacks here
        ],
        max_steps=max_steps,
        verbosity_level=verbosity,
    )
