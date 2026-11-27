"""Type definitions for MissAV topic extraction."""

from typing import Any, List, TypedDict

from plotly.graph_objs import Figure


class VideoRef(TypedDict):
    """Identifies the DB row a document came from."""

    id: str
    video_id: str
    url: str


class VideoRecord(VideoRef):
    """One video group: DB identifiers plus the text used for modeling."""

    code: str
    episode: str
    title: str
    text: str


class DocumentInput(VideoRef):
    """Represents a single input document."""

    index: int
    content: str


class RawInputs(TypedDict):
    """Container for initial loaded documents."""

    total_documents: int
    documents: List[DocumentInput]


class ScoredDocument(VideoRef):
    """Represents a document after n-gram scoring."""

    index: int
    score: float
    ngrams: List[str]
    document: str


class TopicWord(TypedDict):
    """Represents a word within a topic."""

    word: str
    score: float


class TopicDetail(TypedDict):
    """Detailed information about a specific topic."""

    topic_id: int
    representative_document_count: int
    words: List[TopicWord]


class RepresentativeDocument(VideoRef):
    """A document representative of a topic."""

    index: int
    score: float
    content: str


class RepresentativeTopic(TypedDict):
    """Container for representative documents of a topic."""

    topic_id: int
    count: int
    documents: List[RepresentativeDocument]


class DocumentTopicAssignment(VideoRef):
    """Topic assignment for a single document."""

    index: int
    topic_id: int
    topic_name: str
    scores: List[float]


class VisualizationFigures(TypedDict):
    """Container for Plotly figures."""

    barcharts: Figure
    topic_map: Figure
    document_map: Figure


class ClusterTopicsResult(TypedDict):
    """Complete result from the cluster_topics business logic."""

    model: Any
    corpus: List[str]
    corpus_records: List[VideoRecord]
    topic_data: Any
    topics: List[TopicDetail]
    representative_topics: List[RepresentativeTopic]
    document_topics: List[DocumentTopicAssignment]
    figures: VisualizationFigures
    raw_inputs: RawInputs
    preprocessed_corpus: List[str]
    scored_results: List[ScoredDocument]
    sorted_corpus_initial: List[str]
    sorted_records: List[VideoRecord]
