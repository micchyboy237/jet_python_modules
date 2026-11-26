"""Document Clustering with LLM-based Topic Generation.

This module demonstrates clustering a list of documents into thematic groups
and generating descriptive topic strings for each cluster using the LLM utils.
"""

import logging
from typing import Any, Literal

from jet.adapters.llama_cpp.llm_utils import chat
from jet.logger import logger
from pydantic import BaseModel, Field

# Configure logging level for this module
logger.setLevel(logging.DEBUG)

# ─── Type Definitions ────────────────────────────────────────────────────────

Granularity = Literal["loose", "balanced", "tight"]


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
        le=20,
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


# ─── Granularity Configuration ───────────────────────────────────────────────

GRANULARITY_CONFIG: dict[Granularity, dict[str, Any]] = {
    "loose": {
        "cluster_hint_multiplier": 0.6,
        "min_clusters": 2,
        "max_clusters": 8,
        "instruction": (
            "Group documents into BROAD, high-level themes. Prefer fewer, larger clusters. "
            "Only separate documents if they are clearly about completely different subjects. "
            "It is acceptable to group somewhat related topics together."
        ),
    },
    "balanced": {
        "cluster_hint_multiplier": 1.0,
        "min_clusters": 3,
        "max_clusters": 12,
        "instruction": (
            "Group documents into distinct thematic clusters. Balance between broad categories "
            "and specific subtopics. Documents in the same cluster should share a clear common theme."
        ),
    },
    "tight": {
        "cluster_hint_multiplier": 1.5,
        "min_clusters": 4,
        "max_clusters": 20,
        "instruction": (
            "Group documents into VERY SPECIFIC, narrow themes. Prefer more, smaller clusters. "
            "Separate documents even if they are only slightly different in focus or subtopic. "
            "Outliers should be placed in their own individual clusters."
        ),
    },
}


# ─── Clustering Logic ────────────────────────────────────────────────────────


def cluster_documents(
    docs: list[str],
    num_clusters_hint: int = 5,
    granularity: Granularity = "balanced",
) -> dict[int, list[int]]:
    """Cluster documents using an LLM and return a mapping of cluster_id → list of doc indices.

    Args:
        docs: List of document strings to cluster.
        num_clusters_hint: Base suggested number of clusters for the LLM.
        granularity: Controls how loosely or tightly documents are grouped.
            - "loose": Fewer, broader clusters (groups related topics together).
            - "balanced": Moderate number of distinct clusters (default).
            - "tight": More, narrower clusters (separates subtle differences).

    Returns:
        A dictionary mapping cluster IDs (1-based) to lists of document indices.
    """
    config = GRANULARITY_CONFIG[granularity]

    # Adjust cluster hint based on granularity
    adjusted_hint = max(
        config["min_clusters"],
        min(
            config["max_clusters"],
            int(num_clusters_hint * config["cluster_hint_multiplier"]),
        ),
    )

    logger.info(
        f"📄 Starting clustering of {len(docs)} documents "
        f"(hint: ~{adjusted_hint} clusters, granularity: '{granularity}')"
    )

    # Build the prompt with all documents enumerated
    doc_list_str = "\n".join(f"[{i}] {doc}" for i, doc in enumerate(docs))

    system_prompt = (
        f"You are an expert document analyst. Your task is to group the provided documents "
        f"into coherent thematic clusters. Analyze the content of each document carefully "
        f"and assign each one to exactly one cluster. Documents within the same cluster "
        f"should share a common theme or subject matter. Some documents may be outliers "
        f"that do not fit well with others; assign them to their own small clusters if needed. "
        f"Return your answer strictly as valid JSON matching the specified schema.\n\n"
        f"**Clustering Style**: {config['instruction']}"
    )

    user_prompt = (
        f"Here are {len(docs)} documents:\n\n{doc_list_str}\n\n"
        f"Please cluster these documents into approximately {adjusted_hint} thematic groups. "
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


def run_clustering_pipeline(
    docs: list[str],
    num_clusters_hint: int = 5,
    granularity: Granularity = "balanced",
) -> dict[int, dict[str, Any]]:
    """Run the full clustering pipeline: cluster docs → generate topics.

    Args:
        docs: List of document strings.
        num_clusters_hint: Base suggested number of clusters for the LLM.
        granularity: Controls how loosely or tightly documents are grouped.
            - "loose": Fewer, broader clusters.
            - "balanced": Moderate number of distinct clusters (default).
            - "tight": More, narrower clusters.

    Returns:
        A dictionary mapping cluster_id to a dict with keys:
          - 'topic': str
          - 'document_indices': list[int]
          - 'documents': list[str]
    """
    logger.info(
        f"🚀 Starting document clustering pipeline (granularity: '{granularity}')"
    )

    # Step 1: Cluster documents
    cluster_map = cluster_documents(
        docs, num_clusters_hint=num_clusters_hint, granularity=granularity
    )

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
