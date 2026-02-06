#!/usr/bin/env python3
"""
smolagents – demonstration / template code
based on the official guided tour

Last major update pattern: early 2025
"""

from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.libs.smolagents.custom_models import OpenAIModel

# ───────────────────────────────────────────────
#   Choose which model backends you want to support
# ───────────────────────────────────────────────

SUPPORT_INFERENCE_CLIENT = True
SUPPORT_TRANSFORMERS = False
SUPPORT_LITELLM = False  # OpenAI / Anthropic / Ollama / ...
SUPPORT_AZURE_OPENAI = False
SUPPORT_BEDROCK = False
SUPPORT_MLX = False

# Which style do you want to see / use as default?
DEFAULT_AGENT_TYPE = "code"  # "code" or "toolcalling"


# ───────────────────────────────────────────────
#   1. Imports & environment
# ───────────────────────────────────────────────

try:
    from smolagents import (
        AmazonBedrockModel,
        AzureOpenAIModel,
        CodeAgent,
        GradioUI,
        InferenceClientModel,
        LiteLLMModel,
        MLXModel,
        Tool,
        ToolCallingAgent,
        TransformersModel,
        WebSearchTool,
        load_tool,
        tool,
    )
except ImportError as exc:
    print(
        "smolagents not found →  pip install smolagents[toolkit,transformers,litellm,...]"
    )
    raise exc


# Optional – for custom tools
try:
    from huggingface_hub import list_models
except ImportError:
    list_models = None


# ───────────────────────────────────────────────
#   2. Custom tools – two classical styles
# ───────────────────────────────────────────────


@tool
def most_downloaded_model(task: str) -> str:
    """
    Returns the ID (repo name) of the most downloaded model
    for a given task on the Hugging Face Hub.

    Args:
        task: Examples: 'text-classification', 'text-to-image', 'text-to-video', ...
    """
    if list_models is None:
        return "Error: huggingface_hub not installed"

    try:
        m = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return m.id
    except Exception as e:
        return f"Error: {str(e)}"


class MostDownloadedModelTool(Tool):
    name = "most_downloaded_model"
    description = (
        "Returns the repo ID of the most downloaded model on the Hub "
        "for the given task (text-classification, text-generation, ...)"
    )
    inputs = {
        "task": {"type": "string", "description": "The task / tag you are looking for"}
    }
    output_type = "string"

    def forward(self, task: str) -> str:
        if list_models is None:
            return "huggingface_hub not installed"
        try:
            m = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
            return m.id
        except Exception as e:
            return f"Error → {str(e)}"


# ───────────────────────────────────────────────
#   3. Model factory functions
# ───────────────────────────────────────────────


def create_local_model(
    temperature: float = 0.4,
    max_tokens: int | None = 8000,
    model_id: LLAMACPP_LLM_KEYS = "qwen3-instruct-2507:4b",
    agent_name: str | None = None,
) -> OpenAIModel:
    """Factory for creating consistently configured local llama.cpp model."""
    return OpenAIModel(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_name=agent_name,
    )


def create_model() -> object | None:
    return create_local_model()


# ───────────────────────────────────────────────
#   4. Agent factory
# ───────────────────────────────────────────────


def create_agent(
    agent_type="code",
    add_base_tools=True,
    extra_tools=None,
    managed_agents=None,
    **kwargs,
):
    model = create_model()
    if model is None:
        raise RuntimeError("No model available. Check model creation settings.")

    tools = (extra_tools or []).copy()

    if agent_type.lower() == "code":
        klass = CodeAgent
        # CodeAgent can use code → usually no need for PythonInterpreterTool
    else:
        klass = ToolCallingAgent
        # ToolCallingAgent benefits from having PythonInterpreterTool

    return klass(
        tools=tools,
        model=model,
        add_base_tools=add_base_tools,
        managed_agents=managed_agents,
        **kwargs,
    )


# ───────────────────────────────────────────────
#   5. Examples
# ───────────────────────────────────────────────


def example_1_simple_math():
    agent = create_agent(agent_type=DEFAULT_AGENT_TYPE, add_base_tools=True)
    answer = agent.run("What is the 118th Fibonacci number?")
    print("\nFinal answer:", answer)


def example_2_custom_tool():
    agent = create_agent(
        agent_type=DEFAULT_AGENT_TYPE,
        add_base_tools=True,
        extra_tools=[most_downloaded_model],  # or MostDownloadedModelTool()
    )
    answer = agent.run(
        "Which model has the most downloads in the text-to-video task on Hugging Face?"
    )
    print("\nFinal answer:", answer)


def example_3_multi_agent():
    model = create_model()

    web_child = CodeAgent(
        tools=[WebSearchTool()],
        model=model,
        name="web_child",
        description="Performs web searches. Give it a clear search query.",
    )

    manager = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[web_child],
        add_base_tools=True,
    )

    answer = manager.run("Who is currently the CEO of Hugging Face?")
    print("\nFinal answer:", answer)


def example_4_gradio():
    # Usually needs image generation or other interesting tool
    try:
        image_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
    except Exception:
        print("Could not load text-to-image tool → skipping fancy Gradio demo")
        image_tool = None

    agent = create_agent(
        agent_type="code",
        extra_tools=[image_tool] if image_tool else [],
        add_base_tools=True,
    )

    print("Launching Gradio UI ... (open http://127.0.0.1:7860)")
    GradioUI(agent).launch()


# ───────────────────────────────────────────────
#   main
# ───────────────────────────────────────────────


def main():
    print("┌──────────────────────────────────────────────┐")
    print("│           smolagents demo / template         │")
    print("└──────────────────────────────────────────────┘\n")

    print("Choose an example:")
    print("  1) Simple math / Fibonacci")
    print("  2) Custom tool – most downloaded model")
    print("  3) Multi-agent (manager + web search child)")
    print("  4) Gradio UI (interactive)")
    print()

    choice = input("→ ").strip()

    if choice == "1":
        example_1_simple_math()
    elif choice == "2":
        example_2_custom_tool()
    elif choice == "3":
        example_3_multi_agent()
    elif choice == "4":
        example_4_gradio()
    else:
        print("Running default example (custom tool) …")
        example_2_custom_tool()


if __name__ == "__main__":
    main()
