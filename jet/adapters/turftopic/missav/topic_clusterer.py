"""Core business logic for extracting KeyNMF topics from MissAV documents."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from jet.adapters.llama_cpp.chunking_utils import truncate_texts
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.turftopic.chat import LlamaCppAnalyzer
from jet.adapters.turftopic.embed import LlamaCppEncoder
from jet.libs.bertopic.examples.mock import load_missav_videos
from jet.wordnet.keywords.helpers import preprocess_texts
from jet.wordnet.n_grams import score_documents_by_ngrams
from rich.console import Console
from topicwizard.figures.documents import document_map
from topicwizard.figures.topics import topic_barcharts, topic_map
from turftopic import KeyNMF
from turftopic.vectorizers.spacy import NounPhraseCountVectorizer

console = Console()


def cluster_topics(
    code: str = "juc",
    n_topics: int = 5,
    n_docs: int = 100,
    top_k: int = 10,
    no_llm: bool = False,
    representative_top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extracts KeyNMF topics from MissAV documents.

    Args:
        code: MissAV video code prefix.
        n_topics: Number of topics to extract.
        n_docs: Number of documents to process.
        top_k: Number of top words per topic.
        no_llm: Disable LLM-based topic naming.
        representative_top_k: Number of representative documents per topic.

    Returns:
        A dictionary containing:
        - model: The fitted KeyNMF model.
        - corpus: The list of processed documents.
        - topic_data: The TopicData object for visualization.
        - topics: List of topic dictionaries.
        - representative_topics: List of representative document dictionaries.
        - document_topics: List of document-topic assignment dictionaries.
        - figures: Dictionary of plotly figures (barcharts, topic_map, document_map).
        - raw_inputs: Dictionary containing initial loaded documents for reference.
    """
    console.print("📄 Loading documents...")
    docs = load_missav_videos(code=[code])
    docs = truncate_texts(
        docs, model=EMBED_MODEL, max_tokens=128, valid_sentences_only=True
    )
    corpus_initial = [d for d in docs if d.strip()]
    console.print(f"   Loaded {len(corpus_initial)} documents")

    raw_inputs = {
        "total_documents": len(corpus_initial),
        "documents": [
            {"index": i, "content": doc} for i, doc in enumerate(corpus_initial)
        ],
    }

    console.print("Preprocessing and scoring documents...")
    preprocessed_corpus = preprocess_texts(corpus_initial)

    results = score_documents_by_ngrams(
        preprocessed_corpus,
        min_words=1,
        max_words=3,
        min_doc_count=2,
    )

    # Sort by score and limit to n_docs
    sorted_indices = [r["index"] for r in results]
    sorted_corpus = [corpus_initial[idx] for idx in sorted_indices]
    corpus = sorted_corpus[:n_docs]

    console.print(f"Processing {len(corpus)} documents.")

    console.print("Initializing NounPhraseCountVectorizer...")
    vectorizer = NounPhraseCountVectorizer("en_core_web_sm", min_df=1)

    console.print("Initializing LlamaCppEncoder...")
    encoder = LlamaCppEncoder(model=EMBED_MODEL, batch_size=64)

    console.print(f"Fitting KeyNMF model with {n_topics} topics...")
    model = KeyNMF(
        n_components=n_topics,
        vectorizer=vectorizer,
        encoder=encoder,
    )
    document_topic_matrix = model.fit_transform(corpus)

    # Topic Naming
    if not no_llm:
        console.print("🤖 Generating topic names with LLM...")
        try:
            namer = LlamaCppAnalyzer()
            model.rename_topics(namer, use_documents=True)
            console.print("[green]✓ Topics renamed successfully.[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to generate LLM names: {e}[/red]")
            console.print(
                "[yellow]Falling back to default keyword-based names.[/yellow]"
            )
    else:
        console.print("[yellow]⚠ LLM naming disabled via --no-llm flag.[/yellow]")

    # Prepare Representative Documents
    representative_topics = []
    for topic_id, topic_scores in enumerate(np.asarray(document_topic_matrix).T):
        ranked = [
            int(index) for index in np.argsort(-topic_scores) if topic_scores[index] > 0
        ]
        chosen = (
            ranked if representative_top_k is None else ranked[:representative_top_k]
        )
        representative_topics.append(
            {
                "topic_id": topic_id,
                "count": len(chosen),
                "documents": [
                    {
                        "index": index,
                        "score": float(topic_scores[index]),
                        "content": corpus[index],
                    }
                    for index in chosen
                ],
            }
        )

    representative_by_topic = {item["topic_id"]: item for item in representative_topics}

    # Prepare Topic Details
    topics = []
    for topic_id, words in model.get_topics(top_k=top_k):
        topic_id = int(topic_id)
        topics.append(
            {
                "topic_id": topic_id,
                "representative_document_count": representative_by_topic[topic_id][
                    "count"
                ],
                "words": [
                    {"word": word, "score": float(score)} for word, score in words
                ],
            }
        )

    # Prepare Document Assignments
    topic_names = [str(name) for name in model.topic_names]
    document_topics = []
    for index, scores in enumerate(document_topic_matrix):
        top_index = int(scores.argmax())
        document_topics.append(
            {
                "index": index,
                "topic_id": top_index,
                "topic_name": topic_names[top_index],
                "scores": [float(score) for score in scores],
            }
        )

    # Prepare Visualizations
    console.print("Preparing topic visualization...")
    topic_data = model.prepare_topic_data(corpus)
    rep_counts = [item["count"] for item in representative_topics]

    barcharts = topic_barcharts(topic_data)
    for annotation, count in zip(barcharts.layout.annotations, rep_counts):
        annotation.text = f"{annotation.text}<br>({count} representative docs)"

    topic_map_fig = topic_map(topic_data)
    topic_map_fig.update_traces(
        text=[
            f"<b>{name}</b><br>{count} docs"
            for name, count in zip(topic_data["topic_names"], rep_counts)
        ],
        hovertemplate="%{text}<extra></extra>",
        hoverinfo="text",
    )

    represented_by = [[] for _ in corpus]
    for item in representative_topics:
        for document in item["documents"]:
            represented_by[document["index"]].append(str(item["topic_id"]))

    document_metadata = pd.DataFrame(
        {
            "preview": [doc[:160].replace("\n", " ") for doc in corpus],
            "representative_of": [
                ", ".join(topic_ids) if topic_ids else "—"
                for topic_ids in represented_by
            ],
        }
    )
    doc_map_fig = document_map(topic_data, document_metadata=document_metadata)

    figures = {
        "barcharts": barcharts,
        "topic_map": topic_map_fig,
        "document_map": doc_map_fig,
    }

    return {
        "model": model,
        "corpus": corpus,
        "topic_data": topic_data,
        "topics": topics,
        "representative_topics": representative_topics,
        "document_topics": document_topics,
        "figures": figures,
        "raw_inputs": raw_inputs,
        "preprocessed_corpus": preprocessed_corpus,
        "scored_results": results,
        "sorted_corpus_initial": sorted_corpus,
    }
