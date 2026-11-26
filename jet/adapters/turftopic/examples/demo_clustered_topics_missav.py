import shutil
from pathlib import Path

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.turftopic.clustered_topics import LlamaCppEncoder, generate_topics
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    sample_comments = [
        "The video quality is amazing, love the 4K resolution!",
        "Audio is too quiet, can barely hear the speaker",
        "Great tutorial, very helpful for beginners",
        "Please make more videos like this one",
        "The background music is distracting",
        "Excellent explanation of the topic",
        "Video keeps buffering, terrible experience",
        "Love your content, been watching for years",
        "Can you add subtitles? Would help a lot",
        "The lighting in this video is poor",
        "Best channel on YouTube, keep it up!",
        "Timestamps would be really useful",
        "Sound quality needs improvement",
        "Very informative and well-researched",
        "Why is the video so long? Could be shorter",
        "Amazing editing and production quality",
        "Please cover more advanced topics",
        "The intro is too long, get to the point",
        "Subtitles are inaccurate, please fix",
        "Outstanding work as always!",
    ]
    results = generate_topics(
        documents=sample_comments,
        n_topics=5,
        encoder=LlamaCppEncoder(),
        llm_model=LLM_MODEL,
        top_docs_per_topic=3,
        output_dir=OUTPUT_DIR,
    )
    console.print(f"\n[bold]Output directory:[/bold] {results['output_dir']}")
