#!/usr/bin/env python3
"""
smolagents – demonstration / template code
based on the official guided tour

Last major update pattern: early 2025
"""

import os

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


def create_model() -> object | None:
    token = os.getenv("HF_TOKEN") or "<put-your-hf-token-here>"

    if SUPPORT_INFERENCE_CLIENT:
        try:
            return InferenceClientModel(
                model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
                # model_id="meta-llama/Llama-3.3-70B-Instruct",
                token=token,
                temperature=0.3,
                max_tokens=1800,
            )
        except Exception as e:
            print("InferenceClientModel failed →", e)

    if SUPPORT_TRANSFORMERS and SUPPORT_INFERENCE_CLIENT is False:
        try:
            return TransformersModel("meta-llama/Llama-3.2-3B-Instruct")
        except Exception as e:
            print("TransformersModel failed →", e)

    if SUPPORT_LITELLM:
        try:
            return LiteLLMModel(
                model_id="anthropic/claude-3-5-sonnet-latest",
                # model_id="ollama_chat/llama3.1:8b",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                # api_base="http://localhost:11434",   # for ollama
            )
        except Exception as e:
            print("LiteLLMModel failed →", e)

    print("No model could be instantiated.")
    return None


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
