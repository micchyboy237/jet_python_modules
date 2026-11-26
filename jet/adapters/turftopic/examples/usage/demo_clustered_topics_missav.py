import shutil
from pathlib import Path

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.turftopic.clustered_topics import LlamaCppEncoder, generate_topics
from jet.libs.bertopic.examples.mock import load_missav_videos
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    texts, embeddings = load_missav_videos(code=["juc"], include_embeddings=True)
    texts = texts[:20]
    results = generate_topics(
        documents=texts,
        n_topics=5,
        encoder=LlamaCppEncoder(),
        llm_model=LLM_MODEL,
        top_docs_per_topic=3,
        output_dir=OUTPUT_DIR,
    )
    console.print(f"\n[bold]Output directory:[/bold] {results['output_dir']}")
