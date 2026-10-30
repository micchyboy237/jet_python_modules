import numpy as np
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.representation import KeyBERTInspired
from jet.adapters.llama_cpp.config import EMBED_DIMS, EMBED_MODEL

# Import jet adapters for embedding and configuration
from jet.adapters.llama_cpp.embed_utils import embed
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


docs = [
    # --- Cluster 1: Technology & AI (Short & Medium) ---
    "Python is a versatile programming language used in data science.",
    "Machine learning algorithms are transforming healthcare diagnostics.",
    "The new GPU architecture offers significant performance improvements for training large language models.",
    "Cloud computing allows businesses to scale their infrastructure dynamically.",
    "Cybersecurity threats are evolving rapidly with the rise of IoT devices.",
    "Artificial intelligence is not just about automation; it's about augmentation of human capabilities.",
    "The latest smartphone features a revolutionary camera system powered by computational photography.",
    "Quantum computing promises to solve complex problems that are currently intractable for classical computers.",
    "Open-source software has become the backbone of modern web development.",
    "Data privacy regulations like GDPR are changing how companies handle user information.",
    # --- Cluster 2: Finance & Economy (Medium & Long) ---
    "The stock market experienced a significant downturn due to rising interest rates and inflation concerns.",
    "Investors are increasingly looking towards ESG (Environmental, Social, and Governance) funds for sustainable returns.",
    "Cryptocurrency volatility remains a major barrier to its adoption as a stable store of value.",
    "Central banks are carefully balancing inflation control with economic growth stimulation.",
    "Personal finance management apps are helping millennials track their spending and savings goals more effectively.",
    "The housing market shows signs of cooling down after years of unprecedented price increases.",
    "Global supply chain disruptions continue to impact manufacturing costs and consumer prices.",
    "Venture capital funding for startups has decreased significantly in the current economic climate.",
    "Understanding compound interest is crucial for long-term wealth accumulation.",
    "The debate over federal tax policy continues to divide economists and policymakers.",
    # --- Cluster 3: Health & Wellness (Varying Lengths) ---
    "Regular exercise and a balanced diet are key to maintaining good health.",
    "Mental health awareness is growing, leading to more resources for those struggling with anxiety and depression.",
    "Recent studies suggest that intermittent fasting may have benefits for metabolic health.",
    "Sleep hygiene is often overlooked but is critical for cognitive function and overall well-being.",
    "The importance of gut health and the microbiome is becoming a central focus in nutritional science.",
    "Yoga and meditation can significantly reduce stress levels and improve flexibility.",
    "Vaccination remains one of the most effective public health interventions in history.",
    "Chronic inflammation is linked to many diseases, including heart disease and diabetes.",
    "Telemedicine has expanded access to healthcare for rural and underserved populations.",
    "Hydration is essential for physical performance and cognitive clarity.",
    # --- Cluster 4: Sports & Entertainment (Short & Punchy) ---
    "The underdog team won the championship in a stunning upset last night.",
    "Box office records were broken by the latest superhero movie release.",
    "Professional athletes are increasingly using data analytics to improve their performance.",
    "The music festival attracted thousands of fans from around the world.",
    "Esports is now recognized as a legitimate professional sport with massive viewership.",
    "The tennis star announced their retirement after a illustrious career spanning two decades.",
    "Streaming services are changing how we consume television and film content.",
    "Local community sports programs provide vital social connections for youth.",
    "The concert tour sold out in minutes, demonstrating the artist's enduring popularity.",
    "Fantasy sports leagues have become a multi-billion dollar industry.",
    # --- Cluster 5: Cooking & Food (Descriptive) ---
    "Sourdough bread requires patience and practice but yields a delicious, tangy result.",
    "Plant-based diets are gaining popularity for both health and environmental reasons.",
    "The secret to a perfect risotto is slowly adding warm broth while stirring constantly.",
    "Spices from different cultures can transform simple ingredients into exotic dishes.",
    "Food waste is a significant global issue that individuals can help mitigate through better planning.",
    "Fermentation is an ancient technique that preserves food and enhances its nutritional value.",
    "The farm-to-table movement emphasizes fresh, locally sourced ingredients.",
    "Baking is a science where precise measurements are crucial for success.",
    "Street food offers a window into the authentic culinary traditions of a region.",
    "Meal prepping on Sundays can save time and money during the busy work week.",
    # --- Similar/Overlapping Topics (AI Ethics vs General AI) ---
    "Ethical considerations in AI development include bias, fairness, and transparency.",
    "How do we ensure that artificial intelligence systems make fair decisions?",
    "The black box nature of deep learning models raises concerns about accountability.",
    "AI should be designed to augment human decision-making, not replace it entirely.",
    "Regulatory frameworks for AI are still in their infancy globally.",
    # --- Similar/Overlapping Topics (Stock Market vs Personal Finance) ---
    "Diversifying your investment portfolio is a fundamental principle of risk management.",
    "Many people struggle to create a budget that accounts for unexpected expenses.",
    "Retirement planning should start early to take advantage of compounding growth.",
    "Market timing is notoriously difficult, even for professional investors.",
    "Financial literacy education should be mandatory in high schools.",
    # --- Outliers (Random, Nonsensical, or Unique) ---
    "The purple elephant danced on the moon while eating cheese.",
    "Why do we park in driveways and drive on parkways?",
    "A single drop of water holds the reflection of the entire sky.",
    "The quick brown fox jumps over the lazy dog repeatedly.",
    "Silence is the loudest sound in an empty room.",
    "Bananas are berries, but strawberries are not.",
    "The concept of time is relative to the observer's velocity.",
    "I wonder if clouds look down on us and think we are the weird ones.",
    "The number 42 is the answer to life, the universe, and everything.",
    "Sometimes, the best solution is to do nothing at all.",
]

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
