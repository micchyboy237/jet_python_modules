"""
Demonstrates the shuffle feature which introduces variability in results.
Uses weighted sampling + diversity to provide different but still relevant sets.
Uses pre-computed embeddings for efficiency.
"""

import json
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.diversity_utils import (
    compute_shuffle_fetch_k,
    shuffle_and_diversify,
)
from jet.adapters.llama_cpp.embed_utils import embed
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

base_topics = [
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "AI ethics is an important field of study.",
    "Natural language processing helps computers understand text.",
    "Computer vision allows machines to see.",
    "Reinforcement learning is used in robotics.",
    "Generative AI can create images and text.",
    "AI bias is a significant concern in algorithm design.",
    "Transformers revolutionized NLP tasks.",
    "AI safety research focuses on preventing harm.",
    "Expert systems were early forms of AI.",
    "AI is transforming healthcare diagnostics.",
    "Autonomous vehicles rely on AI perception.",
    "AI chips are specialized hardware for neural nets.",
    "Large language models predict next tokens.",
    "AI art generators use diffusion models.",
    "Robotics combines AI with mechanical engineering.",
    "AI in finance detects fraudulent transactions.",
    "Virtual assistants use speech recognition AI.",
    "AI gaming bots learn strategies via self-play.",
    "Medical AI assists in diagnosing diseases.",
    "AI weather prediction models improve accuracy.",
    "Smart homes use AI for energy optimization.",
    "AI copyright laws are currently debated.",
    "Neural networks mimic brain structure.",
    "Supervised learning requires labeled data.",
    "Unsupervised learning finds hidden patterns.",
    "Transfer learning applies knowledge to new tasks.",
    "Federated learning trains models on decentralized data.",
    "Quantum computing may accelerate AI training.",
    "AI sentiment analysis gauges public opinion.",
    "Chatbots use AI for customer service.",
    "AI recommendation engines suggest products.",
    "Computer graphics use AI for upscaling.",
    "AI cybersecurity tools detect anomalies.",
    "Edge AI processes data locally on devices.",
    "AI music composition generates new melodies.",
    "AI writing assistants help draft content.",
    "AI translation services break language barriers.",
    "AI traffic management reduces congestion.",
]

documents = [f"{topic} (ID: {i})" for i, topic in enumerate(base_topics)]

results = [
    {"id": f"ai_doc_{i}", "document": doc, "score": 1.0 - (i * 0.01)}
    for i, doc in enumerate(documents)
]

# Pre-compute embeddings for the entire pool
console.print("[dim]Generating embeddings for candidate pool...[/dim]")
all_embeddings = embed([r["document"] for r in results], show_progress=False)

console.print("[bold blue]=== Shuffle Variety Demo ===[/bold blue]")
console.print(f"Candidate pool size: {len(results)}")
console.print(f"Target top_k: 5")

fetch_k = compute_shuffle_fetch_k(top_k=5)
console.print(f"Fetch K for shuffle: {fetch_k}")
console.print(f"[dim]Note: Using first {fetch_k} items from pool for simulation.[/dim]")

# Slice results and embeddings to simulate the "fetched" pool
candidate_pool = results[:fetch_k]
pool_embeddings = all_embeddings[:fetch_k]

seeds = [42, 123, 999]
all_shuffled_results = {}

for seed in seeds:
    console.print(f"\n[bold yellow]Shuffle Seed: {seed}[/bold yellow]")

    # Pass pre-computed embeddings for the candidate pool
    shuffled = shuffle_and_diversify(
        results=candidate_pool,
        top_k=5,
        diversity=0.6,
        seed=seed,
        relevance_bias=1.0,
        embeddings=pool_embeddings,
    )

    all_shuffled_results[f"seed_{seed}"] = shuffled
    selected_ids = [item["id"] for item in shuffled]
    console.print(f"  Selected IDs: {selected_ids}")

output_file = OUTPUT_DIR / "shuffled_results.json"
output_file.write_text(json.dumps(all_shuffled_results, indent=2))

console.print(f"\n[bold]Saved files:[/]")
for f in sorted(OUTPUT_DIR.iterdir()):
    console.print(f"  📄 [link=file://{f}]{f.name}[/link]")
