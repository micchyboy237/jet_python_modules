import json
import requests
import tiktoken
from typing import Literal, Dict, List, TypedDict, Optional
from transformers import AutoTokenizer
from huggingface_hub import HfApi
from tqdm import tqdm
from jet.transformers.formatters import format_json
from jet.logger import logger
from jet.cache.redis import RedisCache, RedisConfigParams

DEFAULT_SF_EMBED_MODEL = "paraphrase-MiniLM-L12-v2"

# Get the list of ollama models
OLLAMA_MODEL_NAMES = Literal[
    "all-minilm:33m",
    "deepseek-coder-v2:16b-lite-instruct-q3_K_M",
    "deepseek-r1:1.5b-qwen-distill-q4_K_M",
    "deepseek-r1:7b-qwen-distill-q4_K_M",
    "deepseek-r1:8b-0528-qwen3-q4_K_M",
    "embeddinggemma",
    "gemma2:9b-instruct-q4_0",
    "gemma3:4b-it-q4_K_M",
    "gemma3n:e2b-it-q4_K_M",
    "granite3.3:2b",
    "llama3.2",
    "mistral:7b-instruct-v0.3-q3_K_M",
    "mistral:7b-instruct-v0.3-q4_K_M",
    "mxbai-embed-large",
    "nomic-embed-text",
    "nomic-embed-text-v2-moe",
    "qwen2.5-coder:7b-instruct-q4_K_M",
    "qwen2.5vl:3b-q4_K_M",
    "qwen3",
    "qwen3:4b-q4_K_M",
    "qwen3-instruct-2507:4b",
    "theqtcompany/codellama-7b-qml"
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
    "qwen3:4b-q4_K_M",
    "qwen3-instruct-2507:4b",
    "mistral",
    "mistral:7b-instruct-v0.3-q4_K_M",
]

OLLAMA_EMBED_MODELS = Literal[
    "embeddinggemma",
    "mxbai-embed-large",
    "granite-embedding:278m",
    "granite-embedding",
    "paraphrase-multilingual",
    "bge-large",
    "all-minilm:33m",
    "all-minilm:22m",
    "nomic-embed-text",
    "nomic-embed-text-v2-moe",
]


# Map models to context window sizes
OLLAMA_MODEL_CONTEXTS = {
    "all-minilm:33m": 512,
    "deepseek-coder-v2:16b-lite-instruct-q3_K_M": 163840,
    "deepseek-r1:1.5b-qwen-distill-q4_K_M": 131072,
    "deepseek-r1:7b-qwen-distill-q4_K_M": 131072,
    "deepseek-r1:8b-0528-qwen3-q4_K_M": 131072,
    "embeddinggemma": 2048,
    "gemma2:9b-instruct-q4_0": 8192,
    "gemma3:4b-it-q4_K_M": 131072,
    "gemma3n:e2b-it-q4_K_M": 32768,
    "granite3.3:2b": 131072,
    "llama3.2": 131072,
    "mistral:7b-instruct-v0.3-q3_K_M": 32768,
    "mistral:7b-instruct-v0.3-q4_K_M": 32768,
    "mxbai-embed-large": 512,
    "nomic-embed-text": 8192,
    "nomic-embed-text-v2-moe": 2048,
    "qwen2.5-coder:7b-instruct-q4_K_M": 32768,
    "qwen2.5vl:3b-q4_K_M": 128000,
    "qwen3": 40960,
    "qwen3:4b-q4_K_M": 40960,
    "theqtcompany/codellama-7b-qml": 16384,
    "qwen3-instruct-2507:4b": 262144,
}

# Map models to embedding sizes
OLLAMA_MODEL_EMBEDDING_TOKENS = {
    "all-minilm:33m": 384,
    "deepseek-coder-v2:16b-lite-instruct-q3_K_M": 2048,
    "deepseek-r1:1.5b-qwen-distill-q4_K_M": 1536,
    "deepseek-r1:7b-qwen-distill-q4_K_M": 3584,
    "deepseek-r1:8b-0528-qwen3-q4_K_M": 4096,
    "embeddinggemma": 768,
    "gemma2:9b-instruct-q4_0": 3584,
    "gemma3:4b-it-q4_K_M": 2560,
    "gemma3n:e2b-it-q4_K_M": 2048,
    "granite3.3:2b": 2048,
    "llama3.2": 3072,
    "mistral:7b-instruct-v0.3-q3_K_M": 4096,
    "mistral:7b-instruct-v0.3-q4_K_M": 4096,
    "mxbai-embed-large": 1024,
    "nomic-embed-text": 768,
    "nomic-embed-text-v2-moe": 768,
    "qwen2.5-coder:7b-instruct-q4_K_M": 3584,
    "qwen2.5vl:3b-q4_K_M": 2048,
    "qwen3": 4096,
    "qwen3:4b-q4_K_M": 2560,
    "theqtcompany/codellama-7b-qml": 4096,
    "qwen3-instruct-2507:4b": 2560,
}


OLLAMA_HF_MODELS = {
    "all-minilm:33m": "sentence-transformers/all-MiniLM-L12-v2",
    "deepseek-coder-v2:16b-lite-instruct-q3_K_M": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-r1:1.5b-qwen-distill-q4_K_M": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-r1:7b-qwen-distill-q4_K_M": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1:8b-0528-qwen3-q4_K_M": "deepseek-ai/DeepSeek-R1-0528",
    "embeddinggemma": "google/embeddinggemma-300m",
    "gemma2:9b-instruct-q4_0": "google/gemma-2-9b",
    "gemma3:4b-it-q4_K_M": "google/gemma-3-4b-it",
    "gemma3n:e2b-it-q4_K_M": "google/gemma-3n-E2B-it",
    "granite3.3:2b": "ibm-granite/granite-3.3-2b-instruct",
    "llama3.2": "meta-llama/Llama-3.2-3B",
    "mistral:7b-instruct-v0.3-q3_K_M": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral:7b-instruct-v0.3-q4_K_M": "mistralai/Mistral-7B-Instruct-v0.3",
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "nomic-embed-text-v2-moe": "nomic-ai/nomic-embed-text-v2-moe",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
    "qwen2.5-coder:7b-instruct-q4_K_M": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5vl:3b-q4_K_M": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen3": "Qwen/Qwen3-8B",
    "qwen3:4b-q4_K_M": "Qwen/Qwen3-4B",
    "qwen3-instruct-2507:4b": "Qwen/Qwen3-4B-Instruct-2507",
    "theqtcompany/codellama-7b-qml": "QtGroup/CodeLlama-7B-QML",
}

OLLAMA_HF_MODEL_NAMES = Literal[
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-0528",
    "google/embeddinggemma-300m",
    "google/gemma-2-9b",
    "google/gemma-3-4b-it",
    "google/gemma-3n-E2B-it",
    "ibm-granite/granite-3.3-2b-instruct",
    "meta-llama/Llama-3.2-3B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "nomic-ai/nomic-embed-text-v1.5",
    "nomic-ai/nomic-embed-text-v2-moe",
    "mixedbread-ai/mxbai-embed-large-v1",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "QtGroup/CodeLlama-7B-QML",
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


class ModelInfo(TypedDict):
    name: str
    details: Dict


class ModelContext(TypedDict):
    model_name: str
    context_length: int


class ModelEmbedding(TypedDict):
    model_name: str
    embedding_length: int


def build_ollama_model_contexts() -> Dict[str, int]:
    models: List[ModelInfo] = get_local_models()
    ollama_models: Dict[str, int] = {}

    for model in models:
        model_name: str = model["name"]
        details: Dict = get_model_details(model_name)
        model_info: Dict = details.get("model_info", {})
        context_length: int = next(
            (model_info[key] for key in model_info if "context_length" in key),
            0
        )

        clean_model_name: str = model_name.removesuffix(":latest")
        ollama_models[clean_model_name] = context_length

    return dict(sorted(ollama_models.items()))


def build_ollama_model_embeddings() -> Dict[str, int]:
    models: List[ModelInfo] = get_local_models()
    ollama_models: Dict[str, int] = {}

    for model in models:
        model_name: str = model["name"]
        details: Dict = get_model_details(model_name)
        model_info: Dict = details.get("model_info", {})
        embedding_length: int = next(
            (model_info[key] for key in model_info if "embedding_length" in key),
            0
        )

        clean_model_name: str = model_name.removesuffix(":latest")
        ollama_models[clean_model_name] = embedding_length

    return dict(sorted(ollama_models.items()))


def build_ollama_hf_mappings() -> Dict[str, Optional[str]]:
    """
    Build a mapping from Ollama model names to their corresponding Hugging Face model IDs.
    Derives model names from get_local_models and queries Hugging Face Hub.
    
    Returns:
        Dict[str, Optional[str]]: A sorted dictionary mapping Ollama model names to HF model IDs.
        Returns None for models without a matching HF ID.
    """
    mappings: Dict[str, Optional[str]] = {}
    api = HfApi()
    
    # Get model names from get_local_models
    models: List[ModelInfo] = get_local_models()
    
    # Initialize mappings for all Ollama models with progress tracking
    for model in tqdm(models, desc="Mapping Ollama models to HF IDs"):
        ollama_name: str = model["name"]
        clean_model_name: str = ollama_name.removesuffix(":latest")
        mappings[clean_model_name] = None  # Default to None

        # Search Hugging Face Hub for matching model
        search_term = clean_model_name.split(":")[0]  # Remove quantization or variant (e.g., "qwen3:4b" -> "qwen3")
        hf_models = api.list_models(search=search_term, sort="downloads", direction=-1, limit=10)
        
        # Try to find a matching HF model ID
        for hf_model in hf_models:
            hf_id: str = hf_model.id
            mappings[clean_model_name] = hf_id

    return dict(sorted(mappings.items()))


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
