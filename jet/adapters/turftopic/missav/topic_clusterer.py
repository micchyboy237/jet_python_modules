"""Core business logic for extracting KeyNMF topics from MissAV documents."""

from typing import List, Optional

import numpy as np
import pandas as pd
from jet.adapters.llama_cpp.chunking_utils import truncate_texts
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.turftopic.chat import LlamaCppAnalyzer
from jet.adapters.turftopic.embed import LlamaCppEncoder
from jet.wordnet.keywords.helpers import preprocess_texts
from jet.wordnet.n_grams import score_documents_by_ngrams
from rich.console import Console
from topicwizard.figures.documents import document_map
from topicwizard.figures.topics import topic_barcharts, topic_map
from turftopic import KeyNMF
from turftopic.vectorizers.spacy import NounPhraseCountVectorizer

from .services import get_missav_video_records_by_code
from .types import (
    ClusterTopicsResult,
    DocumentTopicAssignment,
    RawInputs,
    RepresentativeDocument,
    RepresentativeTopic,
    ScoredDocument,
    TopicDetail,
    TopicWord,
    VideoRecord,
    VideoRef,
    VisualizationFigures,
)

console = Console()


def _ref(record: VideoRecord) -> VideoRef:
    """Pick only the DB identifiers from a record."""
    return {"id": record["id"], "video_id": record["video_id"], "url": record["url"]}


def cluster_topics(
    code: str = "juc",
    n_topics: int = 5,
    n_docs: int = 100,
    top_k: int = 10,
    no_llm: bool = False,
    representative_top_k: Optional[int] = None,
) -> ClusterTopicsResult:
    """
    Extracts KeyNMF topics from MissAV documents.

    Every document keeps its DB identifiers (id, video_id, url) because the
    text and the identifiers live in the same record for the whole pipeline.

    Args:
        code: MissAV video code prefix.
        n_topics: Number of topics to extract.
        n_docs: Number of documents to process.
        top_k: Number of top words per topic.
        no_llm: Disable LLM-based topic naming.
        representative_top_k: Number of representative documents per topic.

    Returns:
        A ClusterTopicsResult dictionary containing all modeling results and visualizations.
    """
    console.print("📄 Loading documents...")
    records = get_missav_video_records_by_code(code=[code])
    truncated = truncate_texts(
        [r["text"] for r in records],
        model=EMBED_MODEL,
        max_tokens=128,
        valid_sentences_only=True,
    )
    if len(truncated) != len(records):
        raise ValueError(
            f"truncate_texts changed the item count: {len(records)} -> {len(truncated)}"
        )

    # Replace text with the truncated one and drop empty docs (record moves as one unit)
    records_initial: List[VideoRecord] = [
        {**record, "text": text}
        for record, text in zip(records, truncated)
        if text.strip()
    ]
    corpus_initial = [r["text"] for r in records_initial]
    console.print(
        f"   Loaded {len(corpus_initial)} documents "
        f"(dropped {len(records) - len(records_initial)} empty)"
    )

    raw_inputs: RawInputs = {
        "total_documents": len(corpus_initial),
        "documents": [
            {"index": i, **_ref(r), "content": r["text"]}
            for i, r in enumerate(records_initial)
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

    # Sort and cut records (not just texts) so ids stay attached
    sorted_records = [records_initial[r["index"]] for r in results]
    sorted_corpus = [r["text"] for r in sorted_records]
    corpus_records = sorted_records[:n_docs]
    corpus = [r["text"] for r in corpus_records]
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

    representative_topics: List[RepresentativeTopic] = []
    for topic_id, topic_scores in enumerate(np.asarray(document_topic_matrix).T):
        ranked = [
            int(index) for index in np.argsort(-topic_scores) if topic_scores[index] > 0
        ]
        chosen = (
            ranked if representative_top_k is None else ranked[:representative_top_k]
        )
        rep_docs: List[RepresentativeDocument] = [
            {
                "index": index,
                "score": float(topic_scores[index]),
                **_ref(corpus_records[index]),
                "content": corpus_records[index]["text"],
            }
            for index in chosen
        ]
        console.log(
            f"[dim]Topic {topic_id}: {len(rep_docs)} rep docs, "
            f"first video_ids={[d['video_id'] for d in rep_docs[:3]]}[/dim]"
        )
        representative_topics.append(
            {
                "topic_id": topic_id,
                "count": len(chosen),
                "documents": rep_docs,
            }
        )

    representative_by_topic = {item["topic_id"]: item for item in representative_topics}

    topics: List[TopicDetail] = []
    for topic_id, words in model.get_topics(top_k=top_k):
        topic_id = int(topic_id)
        topic_words: List[TopicWord] = [
            {"word": word, "score": float(score)} for word, score in words
        ]
        topics.append(
            {
                "topic_id": topic_id,
                "representative_document_count": representative_by_topic[topic_id][
                    "count"
                ],
                "words": topic_words,
            }
        )

    topic_names = [str(name) for name in model.topic_names]
    document_topics: List[DocumentTopicAssignment] = []
    for index, scores in enumerate(document_topic_matrix):
        top_index = int(scores.argmax())
        document_topics.append(
            {
                "index": index,
                **_ref(corpus_records[index]),
                "topic_id": top_index,
                "topic_name": topic_names[top_index],
                "scores": [float(score) for score in scores],
            }
        )
    console.log(f"[dim]Assigned {len(document_topics)} documents to topics[/dim]")

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
            "video_id": [r["video_id"] for r in corpus_records],
            "preview": [doc[:160].replace("\n", " ") for doc in corpus],
            "representative_of": [
                ", ".join(topic_ids) if topic_ids else "—"
                for topic_ids in represented_by
            ],
        }
    )
    doc_map_fig = document_map(topic_data, document_metadata=document_metadata)

    figures: VisualizationFigures = {
        "barcharts": barcharts,
        "topic_map": topic_map_fig,
        "document_map": doc_map_fig,
    }

    scored_results_with_content: List[ScoredDocument] = []
    for r in results:
        record = records_initial[r["index"]]
        scored_results_with_content.append(
            {**r, **_ref(record), "document": record["text"]}
        )

    return {
        "model": model,
        "corpus": corpus,
        "corpus_records": corpus_records,
        "topic_data": topic_data,
        "topics": topics,
        "representative_topics": representative_topics,
        "document_topics": document_topics,
        "figures": figures,
        "raw_inputs": raw_inputs,
        "preprocessed_corpus": preprocessed_corpus,
        "scored_results": scored_results_with_content,
        "sorted_corpus_initial": sorted_corpus,
        "sorted_records": sorted_records,
    }
