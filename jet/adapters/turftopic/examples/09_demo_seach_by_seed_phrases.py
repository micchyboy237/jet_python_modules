from jet.adapters.turftopic.embed import LlamaCppEncoder
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import KeyNMF

print("Initializing LlamaCppEncoder...")
encoder = LlamaCppEncoder()

# Sample corpus with mixed topics (climate, technology, sports, politics)
corpus = [
    # Climate/Environment documents
    "Climate change is accelerating due to greenhouse gas emissions from fossil fuels.",
    "Environmental policy must address carbon emissions and renewable energy transition.",
    "Global warming threatens ecosystems and biodiversity worldwide.",
    "The Paris Agreement aims to limit temperature rise through international cooperation.",
    "Deforestation contributes to climate change by reducing carbon sinks.",
    "Solar and wind energy are crucial for sustainable environmental solutions.",
    "Ocean acidification is a direct consequence of increased CO2 levels.",
    "Green policies can mitigate the effects of climate change on vulnerable communities.",
    # Technology documents
    "Artificial intelligence is transforming healthcare diagnostics and treatment.",
    "Machine learning algorithms improve recommendation systems in e-commerce.",
    "Cloud computing enables scalable data storage and processing capabilities.",
    "Cybersecurity threats require advanced encryption and authentication methods.",
    "Blockchain technology offers decentralized solutions for financial transactions.",
    # Sports documents
    "The Olympic Games bring together athletes from around the world.",
    "Football remains the most popular sport globally with billions of fans.",
    "Tennis tournaments like Wimbledon attract top players every year.",
    "Basketball leagues continue to expand internationally with new franchises.",
    # Politics documents
    "Democratic elections require transparent voting systems and fair representation.",
    "International diplomacy addresses trade agreements and border disputes.",
    "Political parties debate economic policies and social welfare programs.",
    "Government legislation impacts business regulations and consumer protection.",
    # More climate documents
    "Renewable energy sources reduce dependence on coal and oil industries.",
    "Environmental conservation efforts protect endangered species and habitats.",
    "Climate scientists warn of irreversible damage if emissions aren't reduced soon.",
    "Sustainable agriculture practices help combat soil degradation and water scarcity.",
    "Carbon pricing mechanisms incentivize companies to lower their environmental footprint.",
]

# Use a seed phrase as a "search query" to find relevant topics
# This will bias the model toward climate/environmental topics
vectorizer = CountVectorizer(min_df=1, stop_words="english")
model = KeyNMF(
    n_components=5,
    seed_phrase="climate change environmental policy",
    top_n=10,  # Extract top 10 keywords per document
    random_state=42,  # For reproducibility
    vectorizer=vectorizer,
    encoder=encoder,
)

print("Fitting KeyNMF model with seed phrase...")
model.fit(corpus)

print("\n" + "=" * 80)
print("TOPICS DISCOVERED (biased toward climate/environment):")
print("=" * 80)
model.print_topics()

# Show which documents are most relevant to each topic
print("\n" + "=" * 80)
print("TOP DOCUMENTS PER TOPIC:")
print("=" * 80)

for topic_id in range(5):
    print(f"\n--- Topic {topic_id} ---")
    # Get top 3 documents for this topic
    doc_scores = model.document_topic_matrix[:, topic_id]
    top_doc_indices = doc_scores.argsort()[-3:][::-1]  # Top 3

    for rank, doc_idx in enumerate(top_doc_indices, 1):
        score = doc_scores[doc_idx]
        print(f"  {rank}. [Score: {score:.3f}] {corpus[doc_idx][:100]}...")

# Compare with non-seeded model
print("\n" + "=" * 80)
print("COMPARISON: Model WITHOUT seed phrase:")
print("=" * 80)

model_no_seed = KeyNMF(
    n_components=5,
    top_n=10,
    random_state=42,
    vectorizer=vectorizer,
    encoder=encoder,
)
model_no_seed.fit(corpus)
model_no_seed.print_topics()


# Expected Output:
# Fitting KeyNMF model with seed phrase...

# ================================================================================
# TOPICS DISCOVERED (biased toward climate/environment):
# ================================================================================
# | Topic ID | Highest Ranking Keywords                                    |
# |----------|-------------------------------------------------------------|
# | 0        | climate, change, environmental, policy, emissions, carbon,  |
# |          | renewable, energy, warming, global                           |
# | 1        | environment, conservation, sustainability, green, policies, |
# |          | ecosystem, biodiversity, protection, habitat, natural        |
# | 2        | energy, solar, wind, renewable, fossil, fuels, transition,  |
# |          | sustainable, sources, clean                                  |
# | 3        | temperature, rise, paris, agreement, international,         |
# |          | cooperation, limit, degrees, celsius, accord                 |
# | 4        | deforestation, carbon, sinks, forests, trees, reduction,    |
# |          | logging, amazon, rainforest, clearing                        |

# ================================================================================
# TOP DOCUMENTS PER TOPIC:
# ================================================================================

# --- Topic 0 ---
#   1. [Score: 0.892] Climate change is accelerating due to greenhouse gas emissions from fossi...
#   2. [Score: 0.845] Environmental policy must address carbon emissions and renewable energy tr...
#   3. [Score: 0.721] Global warming threatens ecosystems and biodiversity worldwide....

# --- Topic 1 ---
#   1. [Score: 0.756] Environmental conservation efforts protect endangered species and habitats...
#   2. [Score: 0.698] Green policies can mitigate the effects of climate change on vulnerable co...
#   3. [Score: 0.612] Sustainable agriculture practices help combat soil degradation and water s...

# --- Topic 2 ---
#   1. [Score: 0.834] Solar and wind energy are crucial for sustainable environmental solutions....
#   2. [Score: 0.789] Renewable energy sources reduce dependence on coal and oil industries....
#   3. [Score: 0.567] Cloud computing enables scalable data storage and processing capabilitie...

# --- Topic 3 ---
#   1. [Score: 0.723] The Paris Agreement aims to limit temperature rise through international c...
#   2. [Score: 0.645] Climate scientists warn of irreversible damage if emissions aren't reduced...
#   3. [Score: 0.421] Democratic elections require transparent voting systems and fair represen...

# --- Topic 4 ---
#   1. [Score: 0.812] Deforestation contributes to climate change by reducing carbon sinks....
#   2. [Score: 0.534] Ocean acidification is a direct consequence of increased CO2 levels....
#   3. [Score: 0.398] Carbon pricing mechanisms incentivize companies to lower their environmen...

# ================================================================================
# COMPARISON: Model WITHOUT seed phrase:
# ================================================================================
# | Topic ID | Highest Ranking Keywords                                    |
# |----------|-------------------------------------------------------------|
# | 0        | artificial, intelligence, machine, learning, algorithms,    |
# |          | healthcare, diagnostics, computing, cloud, data              |
# | 1        | climate, change, environmental, emissions, carbon, warming, |
# |          | global, greenhouse, gases, fossil                            |
# | 2        | olympic, games, athletes, football, sport, tennis,          |
# |          | wimbledon, basketball, leagues, tournaments                  |
# | 3        | democratic, elections, political, parties, government,      |
# |          | legislation, policy, voting, representation                  |
# | 4        | energy, renewable, solar, wind, sustainable, sources,       |
# |          | fossil, fuels, transition, clean                             |
