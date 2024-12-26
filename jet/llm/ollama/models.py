import json
import requests
from transformers import AutoTokenizer
from jet.logger import logger
from jet.cache.redis import RedisCache, RedisConfigParams


DEFAULT_CONTEXT_LENGTH = 2048
OLLAMA_MODELS: dict[str, int] = {
    "llama3.1": DEFAULT_CONTEXT_LENGTH,
    "llama3.2": DEFAULT_CONTEXT_LENGTH,
    "mistral": DEFAULT_CONTEXT_LENGTH,
    "codellama": DEFAULT_CONTEXT_LENGTH,
    "qwen2.5-coder": DEFAULT_CONTEXT_LENGTH,
    "deepseek-coder-v2": DEFAULT_CONTEXT_LENGTH,
}

OLLAMA_HF_MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3.1": "meta-llama/Llama-3.1-8B",
    "llama3.2": "meta-llama/Llama-3.2-3B",
    "codellama": "meta-llama/CodeLlama-7b-hf",
    "qwen2.5-coder": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
}

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


# Main function to build the OLLAMA_MODELS dictionary
def build_ollama_models():
    models = get_local_models()
    ollama_models = {}

    for model in models:
        model_name = model["name"]
        details = get_model_details(model_name)
        # Extract context length from the details
        model_info = details.get("model_info", {})
        context_length = [model_info[key] for key in list(
            model_info.keys()) if "context_length" in key][0]

        ollama_models[model_name] = context_length

    return ollama_models


def count_tokens(model_name: str, text: str | list[dict] | list[str], template: str = None) -> int:
    if isinstance(text, list) and all(isinstance(item, dict) and 'role' in item and 'content' in item for item in text):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if not template and tokenizer.chat_template:
            template = tokenizer.chat_template

        model_key = REVERSED_OLLAMA_HF_MODELS[model_name]
        tokenizer.chat_template = template if template else get_chat_template(
            model_key)

        try:
            # Assuming 'apply_chat_template' handles list[dict]
            input_ids = tokenizer.apply_chat_template(text, tokenize=True)
            return len(input_ids)
        except Exception as e:
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if isinstance(text, str):
        tokens = tokenizer.encode(text)
        return len(tokens)
    else:
        tokenized = tokenizer.batch_encode_plus(text, return_tensors=None)
        return sum(len(item) for item in tokenized["input_ids"])


# Run the script and print the result
if __name__ == "__main__":
    OLLAMA_MODELS = build_ollama_models()
    logger.success(json.dumps(OLLAMA_MODELS, indent=2))
