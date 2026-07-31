from typing import Any, Dict, List, Literal, Optional, TypedDict

from jet.adapters.llama_cpp.config import (
    EMBED_BASE_HOST,
    EMBED_BASE_URL,
    LLM_BASE_HOST,
    LLM_BASE_URL,
    RERANK_BASE_HOST,
    RERANK_BASE_URL,
)
from jet.adapters.llama_cpp.models import (
    LLAMACPP_KEYS,
    LLAMACPP_LLM_MODELS,
    LLAMACPP_MODELS,
    LLAMACPP_VALUES,
)
from jet.logger import logger

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
    and rerank host/url config. Prefers *_URL over *_HOST when both are set.
    Deduplicates identical hosts (e.g. all 3 pointing at the same server).

    Returns:
        List[str]: Ordered, deduplicated list of base URLs (no trailing /v1).
                    Falls back to the default localhost URL if none are configured.
    """
    raw_candidates = [
        LLM_BASE_URL or LLM_BASE_HOST,
        EMBED_BASE_URL or EMBED_BASE_HOST,
        RERANK_BASE_URL or RERANK_BASE_HOST,
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
            logger.debug(f"Added candidate host: {normalized}")
        else:
            logger.debug(f"Skipped duplicate host: {normalized}")

    if not urls:
        default_url = get_llama_cpp_base_url()
        logger.debug(f"No hosts configured, falling back to default: {default_url}")
        urls.append(default_url)

    logger.info(f"Resolved {len(urls)} candidate host(s): {urls}")
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
    logger.debug(f"Resolving HF ID for model key: {model_key}")

    # Check LLM models first
    if model_key in LLAMACPP_LLM_MODELS:
        hf_id = LLAMACPP_LLM_MODELS[model_key]
        logger.debug(f"Found in LLM models: {hf_id}")
        return hf_id

    # Check all models (embed, rerank, vision)
    if model_key in LLAMACPP_MODELS:
        hf_id = LLAMACPP_MODELS[model_key]
        logger.debug(f"Found in combined models: {hf_id}")
        return hf_id

    logger.error(f"Model key '{model_key}' not found in any model mapping")
    raise ValueError(f"Model key '{model_key}' not found in LLAMACPP_MODELS")


def determine_model_type(model: Dict[str, Any]) -> ModelType:
    """
    Determine if a model is LLM, Embed, or Rerank based on its status.args.
    Args:
        model: Model dictionary
    Returns:
        ModelType: "llm", "embed", or "rerank"
    """
    args = model.get("status", {}).get("args", [])
    if "--embeddings" in args:
        return "embed"
    elif "--reranking" in args:
        return "rerank"
    else:
        return "llm"


def _fetch_models_from_host(url: str) -> List[Dict[str, Any]]:
    """
    Fetch models from a single host and tag each with model_type.
    Returns an empty list (and logs a warning) if the host is unreachable,
    so one dead host doesn't break the aggregate fetch across all 3.

    Args:
        url: Base URL of the host (no trailing /v1)
    Returns:
        List[Dict[str, Any]]: Raw model dicts with model_type added
    """
    from openai import OpenAI

    client = OpenAI(base_url=f"{url}/v1", api_key="not-needed")
    logger.info(f"Fetching models from {url}")
    try:
        models = client.models.list()
    except Exception as e:
        logger.warning(f"Skipping host {url}, failed to fetch models: {e}")
        return []

    models_data = models.model_dump()["data"]
    logger.debug(f"Retrieved {len(models_data)} model(s) from {url}")

    result = []
    for model in models_data:
        model_type = determine_model_type(model)
        result.append({**model, "model_type": model_type})
    return result


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
                logger.debug(f"Duplicate model id '{model_id}' from {url} skipped")
                continue
            merged[model_id] = model

    logger.info(f"Aggregated {len(merged)} unique model(s) across {len(urls)} host(s)")
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
    Args:
        alias: Model alias or ID
        base_url: Direct URL override (highest priority)
    Returns:
        ModelContextEmbeddingSize: Dict with ctx, ctx_train, and embd_dims
    Raises:
        ValueError: If model is not found
    """
    models = get_models(base_url)
    for model in models["data"]:
        if alias in model.get("aliases", []) or alias == model["id"]:
            meta = model.get("meta", {})
            if not meta:
                raise ValueError(f"No meta data found for model: {alias}")
            return ModelContextEmbeddingSize(
                ctx=meta.get("n_ctx", 0),
                ctx_train=meta.get("n_ctx_train", 0),
                embd_dims=meta.get("n_embd", 0),
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
                    n_embd = ctx_embd_size["embd_dims"]
                except ValueError as e:
                    console.print(
                        f"  [red]❌ Failed to get context/embedding size for {model_id}: {e}[/red]"
                    )
                    n_ctx = "N/A"
                    n_embd = "N/A"

                table.add_row(model_id, model_type, str(n_ctx), str(n_embd), owned_by)

            console.print(table)

    except Exception as e:
        console.print(f"  [red]❌ Failed to fetch models: {e}[/red]")
