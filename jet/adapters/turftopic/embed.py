from typing import Iterable

import numpy as np
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from rich.console import Console
from turftopic.encoders.base import ExternalEncoder

console = Console()


class LlamaCppEncoder(ExternalEncoder):
    """Encoder model using embeddings from a local Llama.cpp embedding server."""

    def __init__(
        self,
        model: str = EMBED_MODEL,
        batch_size: int = 64,
    ):
        self.model = model
        self.batch_size = batch_size
        console.log(f"[dim]Initialized LlamaCppEncoder with model: {self.model}[/dim]")

    def encode(self, sentences: Iterable[str]) -> np.ndarray:
        """Encodes sentences into an embedding matrix using local Llama.cpp server."""
        sentences_list = list(sentences)
        if not sentences_list:
            console.log("[yellow]⚠[/yellow] No sentences to encode.")
            return np.array([])

        console.log(f"[dim]Encoding {len(sentences_list)} sentences...[/dim]")
        try:
            embeddings = embed(
                text=sentences_list,
                model=self.model,
                batch_size=self.batch_size,
                show_progress=True,
                return_format="numpy",
            )
            console.log(f"[dim]Encoded shape: {embeddings.shape}[/dim]")
            return embeddings
        except Exception as e:
            console.print(
                f"[bold red]✗ Error in LlamaCppEncoder.encode:[/bold red] {e}"
            )
            raise
