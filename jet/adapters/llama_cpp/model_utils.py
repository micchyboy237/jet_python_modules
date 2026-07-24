from typing import Any, Dict, List, Literal, Optional, TypedDict

from jet.adapters.llama_cpp.config import LLM_BASE_URL

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


def get_models(base_url: Optional[str] = None) -> ModelsResponse:
    """
    Get all models via OpenAI-compatible /v1/models.
    Args:
        base_url: Direct URL override (highest priority)
    """
    from jet.logger import logger
    from openai import OpenAI

    url = get_llama_cpp_base_url(override=base_url)
    client = OpenAI(base_url=f"{url}/v1", api_key="not-needed")
    logger.info(f"Fetching models from {url}")
    models = client.models.list()
    logger.debug(f"Retrieved {len(models.data)} model(s)")

    # Add model_type to each model
    models_data = models.model_dump()["data"]
    updated_models_data = []
    for model in models_data:
        model_type = determine_model_type(model)
        updated_model = {**model, "model_type": model_type}
        updated_models_data.append(updated_model)

    return {
        "object": models.model_dump()["object"],
        "data": updated_models_data,
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
