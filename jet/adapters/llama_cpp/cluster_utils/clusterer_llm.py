"""Document Clustering with LLM-based Topic Generation.

This module demonstrates clustering a list of documents into thematic groups
and generating descriptive topic strings for each cluster using the LLM utils.
"""

import logging
from typing import Any

from jet.adapters.llama_cpp.llm_utils import chat
from jet.logger import logger
from pydantic import BaseModel, Field

# Configure logging level for this module
logger.setLevel(logging.DEBUG)


# ─── Pydantic Models for Structured Output ───────────────────────────────────


class ClusterAssignment(BaseModel):
    """Represents a single document's assignment to a cluster."""

    document_index: int = Field(
        ..., description="The 0-based index of the document in the input list."
    )
    cluster_id: int = Field(
        ..., description="The cluster ID assigned to this document (1-based)."
    )


class ClusteringResult(BaseModel):
    """Structured output from the LLM clustering step."""

    assignments: list[ClusterAssignment] = Field(
        ...,
        description="List of document-to-cluster assignments.",
        min_length=1,
    )
    num_clusters: int = Field(
        ...,
        ge=2,
        le=10,
        description="Total number of clusters identified.",
    )


class TopicDescription(BaseModel):
    """A human-readable topic string for a single cluster."""

    cluster_id: int = Field(..., description="The cluster ID (1-based).")
    topic: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="A concise, descriptive topic string summarizing the cluster's theme.",
    )


class AllTopicsResult(BaseModel):
    """Structured output containing topic descriptions for all clusters."""

    topics: list[TopicDescription] = Field(
        ...,
        description="List of topic descriptions, one per cluster.",
        min_length=1,
    )


# ─── Clustering Logic ────────────────────────────────────────────────────────


def cluster_documents(
    docs: list[str], num_clusters_hint: int = 5
) -> dict[int, list[int]]:
    """Cluster documents using an LLM and return a mapping of cluster_id → list of doc indices.

    Args:
        docs: List of document strings to cluster.
        num_clusters_hint: Suggested number of clusters for the LLM.

    Returns:
        A dictionary mapping cluster IDs (1-based) to lists of document indices.
    """
    logger.info(
        f"📄 Starting clustering of {len(docs)} documents (hint: ~{num_clusters_hint} clusters)"
    )

    # Build the prompt with all documents enumerated
    doc_list_str = "\n".join(f"[{i}] {doc}" for i, doc in enumerate(docs))

    system_prompt = (
        "You are an expert document analyst. Your task is to group the provided documents "
        "into coherent thematic clusters. Analyze the content of each document carefully "
        "and assign each one to exactly one cluster. Documents within the same cluster "
        "should share a common theme or subject matter. Some documents may be outliers "
        "that do not fit well with others; assign them to their own small clusters if needed. "
        "Return your answer strictly as valid JSON matching the specified schema."
    )

    user_prompt = (
        f"Here are {len(docs)} documents:\n\n{doc_list_str}\n\n"
        f"Please cluster these documents into approximately {num_clusters_hint} thematic groups. "
        f"Assign each document to exactly one cluster. Return a JSON object with 'assignments' "
        f"(a list of objects with 'document_index' and 'cluster_id') and 'num_clusters' "
        f"(the total number of distinct clusters used)."
    )

    logger.debug("🤖 Sending clustering request to LLM...")

    result = chat(
        prompt_or_messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=ClusteringResult,
        temperature=0.2,  # Low temperature for deterministic structured output
        max_tokens=8192,
    )

    if not result.structured or not result.structured.success:
        error_msg = (
            result.structured.error if result.structured else "No structured result"
        )
        logger.error(f"❌ Clustering failed: {error_msg}")
        raise RuntimeError(f"LLM clustering failed: {error_msg}")

    clustering: ClusteringResult = result.structured.parsed
    logger.info(
        f"✅ Clustering complete: {clustering.num_clusters} clusters identified"
    )

    # Convert assignments to cluster_id → list of indices
    cluster_map: dict[int, list[int]] = {}
    for assignment in clustering.assignments:
        cluster_map.setdefault(assignment.cluster_id, []).append(
            assignment.document_index
        )

    # Log cluster sizes
    for cid, indices in sorted(cluster_map.items()):
        logger.debug(f"   Cluster {cid}: {len(indices)} documents → indices {indices}")

    return cluster_map


def generate_topic_strings(
    docs: list[str], cluster_map: dict[int, list[int]]
) -> dict[int, str]:
    """Generate a descriptive topic string for each cluster using an LLM.

    Args:
        docs: The original list of document strings.
        cluster_map: Mapping of cluster_id → list of document indices.

    Returns:
        A dictionary mapping cluster IDs to topic strings.
    """
    logger.info(f"🏷️  Generating topic strings for {len(cluster_map)} clusters...")

    topics: dict[int, str] = {}

    for cluster_id in sorted(cluster_map.keys()):
        indices = cluster_map[cluster_id]
        cluster_docs = [docs[i] for i in indices]

        logger.debug(
            f"   Generating topic for Cluster {cluster_id} ({len(cluster_docs)} docs)"
        )

        doc_snippets = "\n".join(f"- {doc}" for doc in cluster_docs)

        system_prompt = (
            "You are a skilled summarizer. Given a set of documents that belong to the same "
            "thematic cluster, produce a concise, descriptive topic string (5–200 characters) "
            "that captures the essence of the cluster. The topic should be specific enough to "
            "distinguish it from other clusters but general enough to cover all documents."
        )

        user_prompt = (
            f"Documents in Cluster {cluster_id}:\n\n{doc_snippets}\n\n"
            f"Provide a single concise topic string that summarizes the common theme of these documents."
        )

        result = chat(
            prompt_or_messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=TopicDescription,
            temperature=0.3,
            max_tokens=512,
        )

        if not result.structured or not result.structured.success:
            error_msg = (
                result.structured.error if result.structured else "No structured result"
            )
            logger.warning(
                f"⚠️  Topic generation failed for Cluster {cluster_id}: {error_msg}. Using fallback."
            )
            # Fallback: use a generic label
            topics[cluster_id] = f"Cluster {cluster_id} (auto-generated)"
        else:
            topic_desc: TopicDescription = result.structured.parsed
            topics[cluster_id] = topic_desc.topic
            logger.debug(f"   → Topic: '{topic_desc.topic}'")

    return topics


def run_clustering_pipeline(docs: list[str]) -> dict[int, dict[str, Any]]:
    """Run the full clustering pipeline: cluster docs → generate topics.

    Args:
        docs: List of document strings.

    Returns:
        A dictionary mapping cluster_id to a dict with keys:
          - 'topic': str
          - 'document_indices': list[int]
          - 'documents': list[str]
    """
    logger.info("🚀 Starting document clustering pipeline")

    # Step 1: Cluster documents
    cluster_map = cluster_documents(docs)

    # Step 2: Generate topic strings
    topic_map = generate_topic_strings(docs, cluster_map)

    # Step 3: Assemble final result
    result: dict[int, dict[str, Any]] = {}
    for cluster_id in sorted(cluster_map.keys()):
        indices = cluster_map[cluster_id]
        result[cluster_id] = {
            "topic": topic_map.get(cluster_id, "Unknown"),
            "document_indices": indices,
            "documents": [docs[i] for i in indices],
        }

    logger.info("✅ Clustering pipeline complete")
    return result


# ─── Sample Documents ────────────────────────────────────────────────────────
# A diverse set of documents covering multiple themes with some outliers.


documents: list[str] = [
    # --- Cluster 1: Machine Learning & AI ---
    "Deep learning models have revolutionized computer vision tasks such as image classification and object detection.",
    "Transformer architectures like BERT and GPT have become the foundation of modern natural language processing systems.",
    "Reinforcement learning agents can learn optimal policies through trial and error in simulated environments.",
    "Convolutional neural networks excel at extracting spatial hierarchies of features from pixel data.",
    "Transfer learning allows pre-trained models to be fine-tuned on smaller domain-specific datasets efficiently.",
    "Generative adversarial networks pit a generator against a discriminator to produce realistic synthetic data.",
    "Attention mechanisms enable models to weigh the importance of different input tokens dynamically.",
    "Gradient descent optimization algorithms iteratively adjust model weights to minimize loss functions.",
    # --- Cluster 2: Climate Change & Environment ---
    "Rising global temperatures are causing polar ice caps to melt at an unprecedented rate.",
    "Deforestation in the Amazon rainforest contributes significantly to carbon dioxide emissions worldwide.",
    "Renewable energy sources like solar and wind power are becoming cost-competitive with fossil fuels.",
    "Ocean acidification threatens marine ecosystems by reducing the availability of carbonate ions for shell formation.",
    "Extreme weather events such as hurricanes and wildfires are increasing in frequency and intensity due to climate change.",
    "Carbon capture and storage technologies aim to remove CO2 from industrial emissions before it enters the atmosphere.",
    "Biodiversity loss is accelerating as habitats shrink and species face extinction pressures.",
    "The Paris Agreement sets international targets for limiting global warming to well below 2 degrees Celsius.",
    # --- Cluster 3: Web Development & JavaScript ---
    "React components use a virtual DOM to efficiently update the user interface when state changes.",
    "Node.js enables server-side JavaScript execution, allowing full-stack development with a single language.",
    "CSS Grid and Flexbox provide powerful layout systems for building responsive web designs.",
    "TypeScript adds static type checking to JavaScript, catching errors at compile time rather than runtime.",
    "Webpack bundles JavaScript modules and assets into optimized files for production deployment.",
    "RESTful APIs follow standard HTTP methods like GET, POST, PUT, and DELETE for resource manipulation.",
    "Single-page applications load a single HTML page and dynamically update content via AJAX requests.",
    "npm is the default package manager for the JavaScript runtime environment Node.js.",
    # --- Cluster 4: Ancient History & Archaeology ---
    "The Great Pyramid of Giza was constructed during the Fourth Dynasty of ancient Egypt around 2560 BCE.",
    "Roman aqueducts used gravity-fed channels to transport water across vast distances to urban centers.",
    "Cuneiform writing developed in Mesopotamia is considered one of the earliest known systems of written communication.",
    "The Terracotta Army was buried with Qin Shi Huang, the first emperor of China, to protect him in the afterlife.",
    "Stonehenge remains one of the most famous prehistoric monuments, with its purpose still debated by archaeologists.",
    "The Library of Alexandria was a major center of scholarship in the Hellenistic world before its destruction.",
    "Machu Picchu showcases the architectural ingenuity of the Inca civilization high in the Andes Mountains.",
    "Hieroglyphics were deciphered in the 19th century using the Rosetta Stone as a key translation tool.",
    # --- Cluster 5: Nutrition & Health ---
    "Omega-3 fatty acids found in fish support cardiovascular health and reduce inflammation in the body.",
    "Probiotics introduce beneficial bacteria into the gut microbiome, aiding digestion and immune function.",
    "Intermittent fasting restricts eating to specific time windows, potentially improving metabolic health.",
    "Vitamin D deficiency is linked to weakened bone density and increased risk of osteoporosis.",
    "Antioxidants neutralize free radicals that can cause cellular damage and contribute to aging.",
    "The Mediterranean diet emphasizes whole grains, olive oil, fruits, vegetables, and lean proteins.",
    "Hydration is essential for maintaining blood volume, regulating body temperature, and supporting kidney function.",
    "Excessive sugar consumption is associated with insulin resistance and an increased risk of type 2 diabetes.",
    # --- Outliers (should not fit cleanly into any main cluster) ---
    "The stock market experienced significant volatility following the unexpected announcement of interest rate changes.",
    "Quantum entanglement describes a phenomenon where particles remain correlated regardless of the distance separating them.",
    "Impressionist painters like Monet and Renoir focused on capturing the effects of light and movement in their works.",
    "The blockchain ledger provides a decentralized and tamper-resistant record of cryptocurrency transactions.",
]


if __name__ == "__main__":
    # Run the pipeline
    clustered = run_clustering_pipeline(documents)

    # Print results
    print("\n" + "=" * 80)
    print("DOCUMENT CLUSTERING RESULTS")
    print("=" * 80)

    for cluster_id, info in sorted(clustered.items()):
        print(f"\n📁 Cluster {cluster_id}: {info['topic']}")
        print(f"   Documents ({len(info['document_indices'])}):")
        for idx in info["document_indices"]:
            preview = (
                documents[idx][:100] + "..."
                if len(documents[idx]) > 100
                else documents[idx]
            )
            print(f"     [{idx}] {preview}")

    print("\n" + "=" * 80)
