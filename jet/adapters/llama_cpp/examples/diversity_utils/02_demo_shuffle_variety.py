"""
Demonstrates the shuffle feature which introduces variability in results.
Uses weighted sampling + diversity to provide different but still relevant sets.
NOTE: Requires a large candidate pool to demonstrate variability.
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

# Generate a larger pool of documents to ensure sampling occurs
# We need len(results) > sample_size for shuffle to work
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

# Create 40 unique documents
documents = [f"{topic} (ID: {i})" for i, topic in enumerate(base_topics)]

# Simulate a large candidate pool fetched from vector DB
# Scores decrease slightly to simulate relevance ranking
results = [
    {"id": f"ai_doc_{i}", "document": doc, "score": 1.0 - (i * 0.01)}
    for i, doc in enumerate(documents)
]


# Mock embedding function
def mock_get_embeddings(ids):
    id_to_text = {r["id"]: r["document"] for r in results}
    ordered_texts = [id_to_text[i] for i in ids]
    # Only embed what is requested
    return embed(ordered_texts, return_format="numpy", show_progress=False)


console.print("[bold blue]=== Shuffle Variety Demo ===[/bold blue]")
console.print(f"Candidate pool size: {len(results)}")
console.print(f"Target top_k: 5")

# Calculate how many candidates to fetch for shuffle
fetch_k = compute_shuffle_fetch_k(top_k=5)
console.print(f"Fetch K for shuffle: {fetch_k}")
console.print(f"[dim]Note: Using first {fetch_k} items from pool for simulation.[/dim]")

# Perform shuffle and diversify with different seeds
seeds = [42, 123, 999]
all_shuffled_results = {}

for seed in seeds:
    console.print(f"\n[bold yellow]Shuffle Seed: {seed}[/bold yellow]")

    # Use the subset of results that simulates the DB fetch
    candidate_pool = results[:fetch_k]

    shuffled = shuffle_and_diversify(
        results=candidate_pool,
        top_k=5,
        diversity=0.6,
        seed=seed,
        get_embeddings_fn=mock_get_embeddings,
        relevance_bias=1.0,
    )

    all_shuffled_results[f"seed_{seed}"] = shuffled

    selected_ids = [item["id"] for item in shuffled]
    console.print(f"  Selected IDs: {selected_ids}")

# Save output
output_file = OUTPUT_DIR / "shuffled_results.json"
output_file.write_text(json.dumps(all_shuffled_results, indent=2))

console.print(f"\n[bold]Saved files:[/]")
for f in sorted(OUTPUT_DIR.iterdir()):
    console.print(f"  📄 [link=file://{f}]{f.name}[/link]")
