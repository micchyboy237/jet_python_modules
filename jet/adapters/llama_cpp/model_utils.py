from typing import Any, Dict, List, Literal, Optional, TypedDict

from jet.adapters.llama_cpp.config import (
    EMBED_BASE_HOST,
    EMBED_BASE_URL,
    LLM_BASE_HOST,
    LLM_BASE_URL,
    RERANK_BASE_HOST,
    RERANK_BASE_URL,
    VISION_BASE_HOST,
    VISION_BASE_URL,
)
from jet.adapters.llama_cpp.models import (
    LLAMACPP_KEYS,
    LLAMACPP_LLM_MODELS,
    LLAMACPP_MODELS,
    LLAMACPP_VALUES,
)

# Define ModelType as a Literal
ModelType = Literal["llm", "embed", "rerank"]


# Define the structure for the status field
class Status(TypedDict):
    value: Literal["loaded", "unloaded"]
    args: List[str]
    preset: str


# Define the structure for the meta field
class Meta(TypedDict):
    vocab_type: int
    n_vocab: int
    n_ctx: int
    n_ctx_train: int
    n_embd: int
    n_params: int
    size: int
    ftype: str


# Define the structure for the architecture field
class Architecture(TypedDict):
    input_modalities: List[str]
    output_modalities: List[str]


# Define the structure for a model, including model_type
class ModelInfo(TypedDict):
    id: str
    aliases: List[str]
    tags: List[str]
    object: Literal["model"]
    owned_by: str
    created: int
    status: Status
    architecture: Architecture
    source: str
    can_remove: bool
    meta: Optional[Meta]
    model_type: ModelType  # Use ModelType here


# Define the structure for the response from the /models endpoint
class ModelsResponse(TypedDict):
    object: Literal["list"]
    data: List[ModelInfo]


# Define the return type for get_model_ctx_embd_size
class ModelContextEmbeddingSize(TypedDict):
    ctx: int
    ctx_train: int
    embd_dims: int


# --- Functions ---
def get_llama_cpp_base_url(override: Optional[str] = None) -> str:
    """Return base URL for llama.cpp LLM server (no /v1)."""
    if override:
        base = override
    else:
        base = LLM_BASE_URL
    if not base:
        base = "http://localhost:8080"
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


def get_llama_cpp_candidate_urls() -> List[str]:
    """
    Build the list of base URLs to query, sourced from the LLM, embed,
    rerank, and vision host/url config. Prefers *_URL over *_HOST when both are set.
    Deduplicates identical hosts (e.g., all 4 pointing at the same server).
    """
    raw_candidates = [
        LLM_BASE_URL or LLM_BASE_HOST,
        EMBED_BASE_URL or EMBED_BASE_HOST,
        RERANK_BASE_URL or RERANK_BASE_HOST,
        VISION_BASE_URL or VISION_BASE_HOST,
    ]
    seen: set[str] = set()
    urls: List[str] = []
    for candidate in raw_candidates:
        if not candidate:
            continue
        normalized = get_llama_cpp_base_url(override=candidate)
        if normalized not in seen:
            seen.add(normalized)
            urls.append(normalized)
        else:
            pass
    if not urls:
        default_url = get_llama_cpp_base_url()
        urls.append(default_url)
    return urls


def get_model_hf_id(model_key: LLAMACPP_KEYS) -> LLAMACPP_VALUES:
    """
    Convert a llama.cpp model key to its HuggingFace model ID.

    Args:
        model_key: Model key like "llama-3.2:3b" or "nomic-embed:1.5"

    Returns:
        HuggingFace model ID like "meta-llama/Llama-3.2-3B-Instruct"

    Raises:
        ValueError: If model_key is not found in any model mapping
    """

    # Check LLM models first
    if model_key in LLAMACPP_LLM_MODELS:
        hf_id = LLAMACPP_LLM_MODELS[model_key]
        return hf_id

    # Check all models (embed, rerank, vision)
    if model_key in LLAMACPP_MODELS:
        hf_id = LLAMACPP_MODELS[model_key]
        return hf_id

    raise ValueError(f"Model key '{model_key}' not found in LLAMACPP_MODELS")


def determine_model_type(model: Dict[str, Any]) -> ModelType:
    """
    Determine if a model is LLM, Embed, or Rerank based on its tags.

    Priority:
    1. Check model tags for "llm", "embed", or "rerank" (new format)
    2. Fall back to checking status.args for --embeddings/--reranking flags (old format)

    Args:
        model: Model dictionary (normalized or raw)

    Returns:
        ModelType: "llm", "embed", or "rerank"

    Examples:
        >>> determine_model_type({"tags": ["llm"]})
        'llm'
        >>> determine_model_type({"tags": ["rerank"]})
        'rerank'
        >>> determine_model_type({"tags": ["embed"]})
        'embed'
        >>> # Fallback to old format
        >>> determine_model_type({"status": {"args": ["--embeddings"]}})
        'embed'
    """
    # New format: Check tags first
    tags = model.get("tags", [])
    if "embed" in tags:
        return "embed"
    elif "rerank" in tags:
        return "rerank"
    elif "llm" in tags:
        return "llm"

    # Fallback: Old format - check status.args for flags
    args = model.get("status", {}).get("args", [])
    if "--embeddings" in args:
        return "embed"
    elif "--reranking" in args:
        return "rerank"

    # Default to LLM if no type indicators found
    return "llm"


def _fetch_models_from_host(url: str) -> List[Dict[str, Any]]:
    """
    Fetch models from a single host and tag each with model_type.
    Handles both old format ({data: [...]}) and new format ({models: [...], data: [...]}).

    New format priority:
    - The 'data' array contains the loaded model regardless of alias
    - 'meta' in data items provides runtime info (n_ctx, n_embd, etc.)
    - 'models' array provides static model configuration

    Args:
        url: Base URL of the host (no trailing /v1)
    Returns:
        List[Dict[str, Any]]: Raw model dicts with model_type added
    """
    from openai import OpenAI

    client = OpenAI(base_url=f"{url}/v1", api_key="not-needed")

    try:
        response = client.models.list()
    except Exception as e:
        return []

    # Handle both old and new response formats
    try:
        models_dict = response.model_dump()
    except AttributeError:
        # Fallback for dict responses
        models_dict = response if isinstance(response, dict) else response.model_dump()

    # New format has 'models' array (static config) and 'data' array (loaded instances)
    if "data" in models_dict and isinstance(models_dict["data"], list):
        models_data = models_dict["data"]
    # Old format fallback
    elif "models" in models_dict and isinstance(models_dict["models"], list):
        models_data = models_dict["models"]
    else:
        return []

    result = []
    for model in models_data:
        # Normalize model dict based on format
        normalized_model = _normalize_model_dict(model, models_dict)

        # Determine model type from status.args if available
        model_type = determine_model_type(normalized_model)
        normalized_model["model_type"] = model_type

        result.append(normalized_model)

    return result


def _normalize_model_dict(
    model: Dict[str, Any], full_response: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Normalize model dict from new format to match expected ModelInfo structure.

    New format model from 'data' array:
    {
        "id": "path/to/model.gguf",
        "aliases": ["path/to/model.gguf"],
        "tags": [],
        "object": "model",
        "created": 1785935721,
        "owned_by": "llamacpp",
        "meta": {
            "vocab_type": 2,
            "n_vocab": 248320,
            "n_ctx": 4096,
            "n_ctx_train": 262144,
            "n_embd": 2048,
            "n_params": 1881825088,
            "size": 1259846912,
            "ftype": "Q4_K - Medium"
        }
    }
    """
    # If model already has 'status' key, it's in the old format
    if "status" in model:
        return model

    # Model from new format 'data' array
    model_id = model.get("id", "")

    # Find matching model config from 'models' array if available
    model_config = {}
    if "models" in full_response:
        for config_model in full_response["models"]:
            if (
                config_model.get("name") == model_id
                or config_model.get("model") == model_id
            ):
                model_config = config_model
                break

    # Build status from capabilities and configuration
    capabilities = model_config.get("capabilities", [])
    status_value = "loaded"  # Models in 'data' array are loaded

    # Determine status args from capabilities
    status_args = []
    if "completion" in capabilities:
        status_args.extend(["--completion"])
    if "multimodal" in capabilities:
        status_args.extend(["--multimodal"])

    # Build the normalized model dict
    normalized = {
        "id": model_id,
        "aliases": model.get("aliases", [model_id]),
        "tags": model.get("tags", []),
        "object": model.get("object", "model"),
        "owned_by": model.get("owned_by", "llamacpp"),
        "created": model.get("created", 0),
        "status": {
            "value": status_value,
            "args": status_args,
            "preset": model_config.get("details", {}).get("format", ""),
        },
        "architecture": {
            "input_modalities": ["text"]
            + (["image"] if "multimodal" in capabilities else []),
            "output_modalities": ["text"],
        },
        "source": model_config.get("type", "model"),
        "can_remove": False,
        "meta": model.get("meta", {}),
        # Preserve original format info for debugging
        "_format": "new",
        "_model_config": model_config if model_config else None,
    }

    return normalized


def get_models(base_url: Optional[str] = None) -> ModelsResponse:
    """
    Get all models across all configured hosts (LLM, embed, rerank),
    deduplicated by host and by model id.

    Args:
        base_url: Direct URL override (highest priority). When given,
                  only this single host is queried, matching prior behavior
                  for callers that already know exactly which host to hit.
    """
    urls = (
        [get_llama_cpp_base_url(override=base_url)]
        if base_url
        else get_llama_cpp_candidate_urls()
    )

    merged: Dict[str, Dict[str, Any]] = {}
    for url in urls:
        for model in _fetch_models_from_host(url):
            model_id = model["id"]
            if model_id in merged:
                continue
            merged[model_id] = model

    return {
        "object": "list",
        "data": list(merged.values()),
    }


def get_loaded_models(base_url: Optional[str] = None) -> ModelsResponse:
    """
    Get loaded models via OpenAI-compatible /v1/models.
    Args:
        base_url: Direct URL override (highest priority)
    """
    models = get_models(base_url)
    loaded_models = {
        "object": models["object"],
        "data": [
            model
            for model in models["data"]
            if model.get("status", {}).get("value") == "loaded"
        ],
    }
    return loaded_models


def get_all_models_ctx_embd_sizes(
    base_url: Optional[str] = None,
) -> List[ModelContextEmbeddingSize]:
    """
    Get context and embedding dimensions for all loaded models.
    Args:
        base_url: Direct URL override (highest priority)
    Returns:
        List[ModelContextEmbeddingSize]: List of dicts with ctx, ctx_train, and embd_dims
    """
    models = get_models(base_url)
    results = []
    for model in models["data"]:
        meta = model.get("meta")
        if meta:
            results.append(
                ModelContextEmbeddingSize(
                    ctx=meta.get("n_ctx", 0),
                    ctx_train=meta.get("n_ctx_train", 0),
                    embd_dims=meta.get("n_embd", 0),
                )
            )
    return results


def get_model_ctx_embd_size(
    alias: str, base_url: Optional[str] = None
) -> ModelContextEmbeddingSize:
    """
    Get context and embedding dimensions for a model by alias.

    Handles both old and new model formats where:
    - Old format: meta in model.meta
    - New format: meta in data[].meta with runtime values

    Args:
        alias: Model alias or ID
        base_url: Direct URL override (highest priority)
    Returns:
        ModelContextEmbeddingSize: Dict with ctx, ctx_train, and embd_dims
    Raises:
        ValueError: If model is not found or no meta data available
    """
    models = get_models(base_url)

    for model in models["data"]:
        # Check both id and aliases
        if alias in model.get("aliases", []) or alias == model["id"]:
            meta = model.get("meta", {})

            if not meta:
                raise ValueError(f"No meta data found for model: {alias}")

            ctx = meta.get("n_ctx", 0)
            ctx_train = meta.get("n_ctx_train", 0)
            embd_dims = meta.get("n_embd", 0)

            return ModelContextEmbeddingSize(
                ctx=ctx,
                ctx_train=ctx_train,
                embd_dims=embd_dims,
            )

    raise ValueError(f"Model not found: {alias}")


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print(f"[bold blue]{'=' * 60}[/bold blue]")
    console.print("[bold blue]Server: llm[/bold blue]")
    console.print(f"[bold blue]{'=' * 60}[/bold blue]")

    try:
        loaded_models = get_loaded_models()
        model_count = len(loaded_models["data"])

        if model_count == 0:
            console.print("  [yellow]⚠️ No loaded models found[/yellow]")
        else:
            console.print(f"  [green]✅ Found {model_count} loaded model(s):[/green]")

            # Create a table for better readability
            table = Table(
                title="Loaded Models", show_header=True, header_style="bold magenta"
            )
            table.add_column("ID", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Context Size", style="blue")
            table.add_column("Train Context", style="blue")
            table.add_column("Embedding Size", style="blue")
            table.add_column("Owned By", style="white")

            for model in loaded_models["data"]:
                model_id = model["id"]
                model_type = model["model_type"]
                owned_by = model.get("owned_by", "N/A")

                # Get context and embedding size
                try:
                    ctx_embd_size = get_model_ctx_embd_size(model_id)
                    n_ctx = ctx_embd_size["ctx"]
                    n_ctx_train = ctx_embd_size["ctx_train"]
                    n_embd = ctx_embd_size["embd_dims"]
                except ValueError as e:
                    console.print(
                        f"  [red]❌ Failed to get context/embedding size for {model_id}: {e}[/red]"
                    )
                    n_ctx = "N/A"
                    n_ctx_train = "N/A"
                    n_embd = "N/A"

                table.add_row(
                    model_id,
                    model_type,
                    str(n_ctx),
                    str(n_ctx_train),
                    str(n_embd),
                    owned_by,
                )

            console.print(table)

    except Exception as e:
        console.print(f"  [red]❌ Failed to fetch models: {e}[/red]")
