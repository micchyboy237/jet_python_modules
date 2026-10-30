import numpy as np
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.representation import KeyBERTInspired
from jet.adapters.llama_cpp.config import EMBED_DIMS, EMBED_MODEL

# Import jet adapters for embedding and configuration
from jet.adapters.llama_cpp.embed_utils import embed
from jet.libs.bertopic.examples.doc_samples import DOCS_LG
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class JetEmbedder(BaseEmbedder):
    """Custom BERTopic backend that uses jet.adapters for embeddings."""

    def __init__(self, model_name: str = EMBED_MODEL, dims: int = EMBED_DIMS):
        super().__init__()
        self.model_name = model_name
        self.embedding_size = dims

    def embed_documents(self, documents, verbose=False):
        # Use the jet adapter's embed function which handles batching and progress
        # We disable show_progress here because BERTopic has its own logging/progress
        return embed(documents, model=self.model_name, show_progress=False)

    def embed_query(self, query):
        # embed() handles both str and list[str]
        result = embed(query, model=self.model_name, show_progress=False)
        # Ensure it returns a 2D array if input was a single string,
        # though embed usually returns 1D for str. BERTopic expects 2D for docs.
        if isinstance(query, str):
            return result.reshape(1, -1)
        return result


docs = DOCS_LG

console.print(
    Panel.fit(
        "[bold green]Starting BERTopic Analysis with KeyBERTInspired & Jet Adapters[/bold green]"
    )
)

# 1. Initialize Custom Backend
jet_embedder = JetEmbedder()

# 2. Generate Initial Embeddings using Jet Adapters
console.print(
    f"\n[yellow]Generating initial embeddings using model: {EMBED_MODEL}...[/yellow]"
)
embeddings = embed(docs, model=EMBED_MODEL, show_progress=True)

# 3. Configure Representation Model
representation_model = KeyBERTInspired(
    top_n_words=5,
    nr_repr_docs=3,
    nr_samples=10,
    nr_candidate_words=20,
)

# 4. Initialize BERTopic
# We pass the custom jet_embedder so KeyBERTInspired can use it internally
topic_model = BERTopic(
    min_topic_size=2,
    calculate_probabilities=True,
    representation_model=representation_model,
    embedding_model=jet_embedder,
    verbose=True,
)

console.print("\n[yellow]Fitting topic model...[/yellow]")
# We can still pass pre-computed embeddings to save time on the first pass
topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

# 5. Display Results with Scoring Info

# --- A. Topic Keywords with Scores ---
table_keywords = Table(title="Topic Keywords & Scores (KeyBERT Inspired)")
table_keywords.add_column("Topic ID", style="cyan", justify="right")
table_keywords.add_column("Keywords", style="magenta")
table_keywords.add_column("Scores (Cosine Sim)", style="green")

for topic_id in sorted(topic_model.get_topics().keys()):
    if topic_id == -1:
        continue

    keywords_scores = topic_model.get_topic(topic_id)
    keywords_str = ", ".join([word for word, _ in keywords_scores])
    scores_str = ", ".join([f"{score:.4f}" for _, score in keywords_scores])

    table_keywords.add_row(str(topic_id), keywords_str, scores_str)

console.print(table_keywords)

# --- B. Representative Documents ---
console.print("\n[bold blue]Representative Documents per Topic[/bold blue]")

repr_docs = topic_model.get_representative_docs()
for topic_id, docs_list in repr_docs.items():
    if topic_id == -1 or not docs_list:
        continue

    panel_content = ""
    for i, doc in enumerate(docs_list):
        # Truncate long docs for display
        display_doc = doc[:80] + "..." if len(doc) > 80 else doc
        panel_content += f"[{i + 1}] {display_doc}\n"

    console.print(Panel(panel_content, title=f"Topic {topic_id}", border_style="blue"))

# --- C. Document Probabilities (Scoring Info) ---
console.print("\n[bold yellow]Document Probability Samples[/bold yellow]")

table_probs = Table(title="Document Assignment Probabilities")
table_probs.add_column("Doc Index", style="cyan")
table_probs.add_column("Assigned Topic", style="magenta")
table_probs.add_column("Probability", style="green")
table_probs.add_column("Top 3 Topic Probs", style="dim")

# Show first 5 documents
for i in range(min(5, len(docs))):
    assigned_topic = topics[i]
    prob_assigned = probs[i][assigned_topic] if assigned_topic != -1 else 0.0

    # Get top 3 probabilities for this document
    top_indices = np.argsort(probs[i])[::-1][:3]
    top_probs = [(idx, probs[i][idx]) for idx in top_indices]
    top_probs_str = ", ".join([f"T{idx}:{p:.2f}" for idx, p in top_probs])

    table_probs.add_row(
        str(i), str(assigned_topic), f"{prob_assigned:.4f}", top_probs_str
    )

console.print(table_probs)

# --- D. Topic Info Summary ---
console.print("\n[bold white]Topic Info Summary[/bold white]")
topic_info = topic_model.get_topic_info()
console.print(topic_info[["Topic", "Count", "Name", "Representation"]].head())
