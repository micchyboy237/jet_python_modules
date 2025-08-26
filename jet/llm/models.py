from typing import Literal
from enum import Enum
import json
from jet.transformers.formatters import format_json
import requests
import tiktoken
from transformers import AutoTokenizer
from jet.logger import logger
from jet.cache.redis import RedisCache, RedisConfigParams

DEFAULT_SF_EMBED_MODEL = "paraphrase-MiniLM-L12-v2"

# Get the list of ollama models
OLLAMA_MODEL_NAMES = Literal[
    "llava",
    "theqtcompany/codellama-7b-qml",
    "qwen3",
    "mxbai-embed-large",
    "gemma3",
    "deepseek-r1",
    "llama3.2",
    "qwen3:4b",
    "gemma3:1b-it-qat",
    "deepseek-r1:1.5b",
    "llama3.2:1b",
    "granite-embedding:278m",
    "granite-embedding",
    "paraphrase-multilingual",
    "bge-large",
    "all-minilm:33m",
    "all-minilm:22m"
]

OLLAMA_LLM_MODELS = Literal[
    "llava",
    "theqtcompany/codellama-7b-qml",
    "qwen3",
    "gemma3",
    "deepseek-r1",
    "llama3.2",
    "qwen3:4b",
    "gemma3:1b-it-qat",
    "deepseek-r1:1.5b",
    "llama3.2:1b",
]

OLLAMA_EMBED_MODELS = Literal[
    "mxbai-embed-large",
    "granite-embedding:278m",
    "granite-embedding",
    "paraphrase-multilingual",
    "bge-large",
    "all-minilm:33m",
    "all-minilm:22m",
]


# Map models to context window sizes
OLLAMA_MODEL_CONTEXTS = {
    "llava": 32768,
    "theqtcompany/codellama-7b-qml": 16384,
    "qwen3": 40960,
    "mxbai-embed-large": 512,
    "gemma3": 131072,
    "deepseek-r1": 131072,
    "llama3.2": 131072,
    "qwen3:4b": 40960,
    "gemma3:1b-it-qat": 32768,
    "deepseek-r1:1.5b": 131072,
    "llama3.2:1b": 131072,
    "granite-embedding:278m": 512,
    "granite-embedding": 512,
    "paraphrase-multilingual": 512,
    "bge-large": 512,
    "all-minilm:33m": 512,
    "all-minilm:22m": 512
}

# Map models to embedding sizes
OLLAMA_MODEL_EMBEDDING_TOKENS = {
    "llava": 4096,
    "theqtcompany/codellama-7b-qml": 4096,
    "qwen3": 4096,
    "mxbai-embed-large": 1024,
    "gemma3": 2560,
    "deepseek-r1": 4096,
    "llama3.2": 3072,
    "qwen3:4b": 2560,
    "gemma3:1b-it-qat": 1152,
    "deepseek-r1:1.5b": 1536,
    "llama3.2:1b": 2048,
    "granite-embedding:278m": 768,
    "granite-embedding": 384,
    "paraphrase-multilingual": 768,
    "bge-large": 1024,
    "all-minilm:33m": 384,
    "all-minilm:22m": 384
}


OLLAMA_HF_MODELS = {
    # LLM
    "llama3.1": "meta-llama/Llama-3.1-8B",
    "llama3.2": "meta-llama/Llama-3.2-3B",
    "llama3.2:1b": "meta-llama/Llama-3.2-1B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "deepseek-r1:1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "gemma3:1b": "google/gemma-3-1b-it",
    "gemma3:4b": "google/gemma-3-4b-it",
    "codegemma": "google/codegemma-7b",
    "codellama": "meta-llama/CodeLlama-7b-hf",
    "qwen3": "Qwen/Qwen3-8B",
    "qwen3:4b": "Qwen/Qwen3-4B",
    "qwen2.5-coder": "Qwen/Qwen2.5-Coder-7B-Instruct",
    # Embed
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
    "granite-embedding": "ibm-granite/granite-embedding-30m-english",
    "granite-embedding:278m": "ibm-granite/granite-embedding-278m-multilingual",
    "all-minilm:22m": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm:33m": "sentence-transformers/all-MiniLM-L12-v2",
    "snowflake-arctic-embed:33m": "Snowflake/snowflake-arctic-embed-s",
    "snowflake-arctic-embed:137m": "Snowflake/snowflake-arctic-embed-m-long",
    "snowflake-arctic-embed": "Snowflake/snowflake-arctic-embed-l",
    "paraphrase-multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bge-large": "BAAI/bge-large-en-v1.5",
}

OLLAMA_HF_MODEL_NAMES = Literal[
    # LLM
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-1B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/codegemma-7b",
    "meta-llama/CodeLlama-7b-hf",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B",
    # Embed
    "nomic-ai/nomic-embed-text-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1",
    "ibm-granite/granite-embedding-30m-english",
    "ibm-granite/granite-embedding-278m-multilingual",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "Snowflake/snowflake-arctic-embed-s",
    "Snowflake/snowflake-arctic-embed-m-long",
    "Snowflake/snowflake-arctic-embed-l",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "BAAI/bge-large-en-v1.5",
]

OLLAMA_HF_MODEL_CHAT_TEMPLATES = {}

# Reverse the dictionary
REVERSED_OLLAMA_HF_MODELS = {v: k for k, v in OLLAMA_HF_MODELS.items()}

# Define the base URL for the API
BASE_URL = "http://localhost:11434/api"
DEFAULT_CONFIG: RedisConfigParams = {
    "host": "localhost",
    "port": 3103,
    "db": 0,
    "max_connections": 100
}


# Function to get the list of available models
def get_local_models():
    cache = RedisCache(config=DEFAULT_CONFIG)
    cache_key = "get_local_models: " + json.dumps(OLLAMA_HF_MODELS)
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result

    logger.info(f"get_local_models: Cache miss for {cache_key}")
    response = requests.get(f"{BASE_URL}/tags")
    response.raise_for_status()  # Raise an exception if the request failed

    result = response.json().get("models", [])
    if result:
        cache.set(cache_key, result)
    return result


def get_model_details(model_name: str) -> str:
    cache = RedisCache(config=DEFAULT_CONFIG)
    cache_key = "get_model_details: " + model_name
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result

    logger.info(f"get_model_details: Cache miss for {cache_key}")

    # Modify to use POST request with "name" in body
    response = requests.post(f"{BASE_URL}/show", json={"name": model_name})
    response.raise_for_status()  # Raise an exception if the request failed

    result = response.json()
    if result:
        cache.set(cache_key, result)
    return result


def get_token_max_length(model_name: str) -> int:
    details = get_model_details(model_name)
    model_info = details.get("model_info", {})
    context_length = [model_info[key] for key in list(
        model_info.keys()) if "context_length" in key][0]
    return context_length


def get_chat_template(model_name: str) -> str:
    model_info = get_model_details(model_name)
    result = model_info.get("template", "")
    return result


# Main function to build the OLLAMA_MODEL_CONTEXTS dictionary
def build_ollama_model_contexts():
    models = get_local_models()
    ollama_models = {}

    for model in models:
        model_name = model["name"]
        details = get_model_details(model_name)
        # Extract context length from the details
        model_info = details.get("model_info", {})
        context_length = [model_info[key] for key in list(
            model_info.keys()) if "context_length" in key][0]

        clean_model_name = model_name.removesuffix(":latest")
        ollama_models[clean_model_name] = context_length

    return ollama_models


def build_ollama_model_embeddings():
    models = get_local_models()
    ollama_models = {}

    for model in models:
        model_name = model["name"]
        details = get_model_details(model_name)
        # Extract context length from the details
        model_info = details.get("model_info", {})
        embedding_length = [model_info[key] for key in list(
            model_info.keys()) if "embedding_length" in key][0]

        clean_model_name = model_name.removesuffix(":latest")
        ollama_models[clean_model_name] = embedding_length

    return ollama_models


def count_tokens(model_name: str, text: str | list[dict] | list[str], template: str = None) -> int:
    if isinstance(text, list) and all(isinstance(item, dict) and 'role' in item and 'content' in item for item in text):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            if not template and tokenizer.chat_template:
                template = tokenizer.chat_template

            model_key = REVERSED_OLLAMA_HF_MODELS[model_name]
            tokenizer.chat_template = template if template else get_chat_template(
                model_key)

            # Assuming 'apply_chat_template' handles list[dict]
            input_ids = tokenizer.apply_chat_template(text, tokenize=True)
            return len(input_ids)
        except Exception as e:
            if isinstance(e, OSError):
                logger.orange(e)
            # logger.newline()
            # logger.debug(model_name)
            # logger.error("Error on template:")
            # logger.error(e)
            # logger.warning(tokenizer.chat_template)
            texts = [str(item) for item in text]
            return count_encoded_tokens(model_name, texts)
    else:
        return count_encoded_tokens(model_name, text)


def count_encoded_tokens(model_name: str, text: str | list[str]) -> int:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(str(text)))

    if isinstance(text, str):
        tokens = tokenizer.encode(text)
        return len(tokens)
    else:
        tokenized = tokenizer.batch_encode_plus(text, return_tensors=None)
        return sum(len(item) for item in tokenized["input_ids"])


# Run the script and print the result
if __name__ == "__main__":
    ollama_model_contexts = build_ollama_model_contexts()
    ollama_model_embeddings = build_ollama_model_embeddings()
    ollama_model_names = list(ollama_model_contexts.keys())

    logger.newline()
    logger.debug(f"ollama_model_contexts ({len(ollama_model_contexts)})")
    logger.success(format_json(ollama_model_contexts))

    logger.newline()
    logger.debug(f"ollama_model_embeddings ({len(ollama_model_embeddings)})")
    logger.success(format_json(ollama_model_embeddings))

    logger.newline()
    logger.debug(f"ollama_model_names ({len(ollama_model_names)})")
    logger.success(format_json(ollama_model_names))
