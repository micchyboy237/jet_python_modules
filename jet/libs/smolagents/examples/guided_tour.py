model = InferenceClientModel()
agent = CodeAgent(
    tools=[], model=model, additional_authorized_imports=["requests", "bs4"]
)
agent.run(
    "Could you get me the title of the page at url 'https://huggingface.co/blog'?"
)

from smolagents import ToolCallingAgent, WebSearchTool

agent = ToolCallingAgent(tools=[WebSearchTool()], model=model)
agent.run(
    "Could you get me the title of the page at url 'https://huggingface.co/blog'?"
)

from smolagents import CodeAgent, InferenceClientModel

model_id = "meta-llama/Llama-3.3-70B-Instruct"

model = InferenceClientModel(
    model_id=model_id, token="<YOUR_HUGGINGFACEHUB_API_TOKEN>"
)  # You can choose to not pass any model_id to InferenceClientModel to use a default model
# you can also specify a particular provider e.g. provider="together" or provider="sambanova"
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)

# !pip install 'smolagents[transformers]'
from smolagents import CodeAgent, TransformersModel

model_id = "meta-llama/Llama-3.2-3B-Instruct"

model = TransformersModel(model_id=model_id)
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)

# !pip install 'smolagents[litellm]'
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(
    model_id="anthropic/claude-3-5-sonnet-latest", api_key="YOUR_ANTHROPIC_API_KEY"
)  # Could use 'gpt-4o'
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)

# !pip install 'smolagents[litellm]'
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(
    model_id="ollama_chat/llama3.2",  # This model is a bit weak for agentic behaviours though
    api_base="http://localhost:11434",  # replace with 127.0.0.1:11434 or remote open-ai compatible server if necessary
    api_key="YOUR_API_KEY",  # replace with API key if necessary
    num_ctx=8192,  # ollama default is 2048 which will fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
)

agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)

# !pip install 'smolagents[openai]'
from smolagents import AzureOpenAIModel, CodeAgent

model = AzureOpenAIModel(model_id="gpt-4o-mini")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)

import os

from smolagents import CodeAgent, LiteLLMModel

AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = (
    "gpt-35-turbo-16k-deployment"  # example of deployment name
)

os.environ["AZURE_API_KEY"] = ""  # api_key
os.environ["AZURE_API_BASE"] = ""  # "https://example-endpoint.openai.azure.com"
os.environ["AZURE_API_VERSION"] = ""  # "2024-10-01-preview"

model = LiteLLMModel(model_id="azure/" + AZURE_OPENAI_CHAT_DEPLOYMENT_NAME)
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)

# !pip install 'smolagents[bedrock]'
from smolagents import AmazonBedrockModel, CodeAgent

model = AmazonBedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)

import boto3
from smolagents import AmazonBedrockModel

# Create a custom Bedrock client
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
)

additional_api_config = {
    "inferenceConfig": {"maxTokens": 3000},
    "guardrailConfig": {"guardrailIdentifier": "identify1", "guardrailVersion": "v1"},
}

# Initialize with comprehensive configuration
model = AmazonBedrockModel(
    model_id="us.amazon.nova-pro-v1:0",
    client=bedrock_client,  # Use custom client
    **additional_api_config,
)

agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)

from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(model_name="bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
agent = CodeAgent(tools=[], model=model)

agent.run("Explain the concept of quantum computing")

# !pip install 'smolagents[mlx-lm]'
from smolagents import CodeAgent, MLXModel

mlx_model = MLXModel("mlx-community/Qwen2.5-Coder-32B-Instruct-4bit")
agent = CodeAgent(model=mlx_model, tools=[], add_base_tools=True)

agent.run("Could you give me the 118th number in the Fibonacci sequence?")

from smolagents import REMOVE_PARAMETER, OpenAIModel

# Remove "stop" parameter
model = OpenAIModel(
    model_id="gpt-5",
    stop=REMOVE_PARAMETER,  # Ensures "stop" is not included in API calls
    temperature=0.7,
)

agent = CodeAgent(tools=[], model=model, add_base_tools=True)

from smolagents import CodeAgent, InferenceClientModel


# Define a custom final answer check function
def is_integer(final_answer: str, agent_memory=None) -> bool:
    """Return True if final_answer is an integer."""
    try:
        int(final_answer)
        return True
    except ValueError:
        return False


# Initialize agent with custom final answer check
agent = CodeAgent(
    tools=[], model=InferenceClientModel(), final_answer_checks=[is_integer]
)

agent.run("Calculate the least common multiple of 3 and 7")

# !pip install 'smolagents[toolkit]'
from smolagents import WebSearchTool

search_tool = WebSearchTool()
print(search_tool("Who's the current president of Russia?"))

from huggingface_hub import list_models

task = "text-classification"

most_downloaded_model = next(
    iter(list_models(filter=task, sort="downloads", direction=-1))
)
print(most_downloaded_model.id)

from smolagents import tool


@tool
def model_download_tool(task: str) -> str:
    """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint.

    Args:
        task: The task for which to get the download count.
    """
    most_downloaded_model = next(
        iter(list_models(filter=task, sort="downloads", direction=-1))
    )
    return most_downloaded_model.id


from smolagents import CodeAgent, InferenceClientModel

agent = CodeAgent(tools=[model_download_tool], model=InferenceClientModel())
agent.run(
    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
)

from smolagents import Tool


class ModelDownloadTool(Tool):
    name = "model_download_tool"
    description = "This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub. It returns the name of the checkpoint."
    inputs = {
        "task": {
            "type": "string",
            "description": "The task for which to get the download count.",
        }
    }
    output_type = "string"

    def forward(self, task: str) -> str:
        most_downloaded_model = next(
            iter(list_models(filter=task, sort="downloads", direction=-1))
        )
        return most_downloaded_model.id


from smolagents import CodeAgent, InferenceClientModel

agent = CodeAgent(tools=[ModelDownloadTool()], model=InferenceClientModel())
agent.run(
    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
)

from smolagents import CodeAgent, InferenceClientModel, WebSearchTool

model = InferenceClientModel()

web_agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    name="web_search_agent",
    description="Runs web searches for you. Give it your query as an argument.",
)

manager_agent = CodeAgent(tools=[], model=model, managed_agents=[web_agent])

manager_agent.run("Who is the CEO of Hugging Face?")

from smolagents import CodeAgent, GradioUI, InferenceClientModel, load_tool

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

model = InferenceClientModel(model_id=model_id)

# Initialize the agent with the image generation tool
agent = CodeAgent(tools=[image_generation_tool], model=model)

GradioUI(agent).launch()

agent.push_to_hub("m-ric/my_agent")

agent.from_hub("m-ric/my_agent", trust_remote_code=True)
