import xml.etree.ElementTree as ET
import markdown
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor
from markdown.extensions.toc import TocExtension
import re
import numpy as np
from typing import List, Dict, Optional, TypedDict, Literal
import logging
from rank_bm25 import BM25Okapi
from sklearn.metrics import ndcg_score

from jet.models.embeddings.base import generate_embeddings
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentChunk(TypedDict):
    id: str
    text: str
    metadata: Dict[str, str]


class SearchResult(TypedDict):
    chunk: DocumentChunk
    score: float


class VectorSearch:
    def __init__(self, embedding_model: EmbedModelType = "all-MiniLM-L12-v2"):
        """Initialize vector search with embedding model."""
        self.model = SentenceTransformerRegistry.load_model(
            embedding_model, device="mps")  # Optimized for Mac M1
        self.chunks: List[DocumentChunk] = []
        self.embeddings: List[np.ndarray] = []  # Store embeddings separately
        self.bm25 = None

    def preprocess_and_index(self, documents: List[Dict[str, str]], chunk_size: int = 500) -> None:
        """Preprocess documents, create chunks, and index embeddings."""
        self.chunks = []
        self.embeddings = []
        for doc in documents:
            logger.debug(f"Processing document: {doc['id']}")
            chunks = self._dynamic_chunking(doc["text"], chunk_size)
            logger.debug(f"Generated chunks: {chunks}")
            chunk_embeddings = generate_embeddings(
                chunks, self.model, return_format="numpy", show_progress=True)
            logger.debug(f"Embeddings shape: {chunk_embeddings.shape}")
            for i, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                chunk_id = f"{doc['id']}_{i}"
                chunk: DocumentChunk = {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": doc["metadata"]
                }
                self.chunks.append(chunk)
                self.embeddings.append(embedding)
                logger.debug(f"Indexed chunk {chunk_id}: {chunk_text}")
        tokenized_chunks = [chunk["text"].split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        logger.info(f"Indexed {len(self.chunks)} chunks")

    def _dynamic_chunking(self, text: str, base_chunk_size: int) -> List[str]:
        """Dynamically chunk text based on semantic boundaries."""
        logger.debug(f"Input text: {text}")
        logger.debug(f"Base chunk size: {base_chunk_size}")
        sentences = re.split(r'(?<=\w\.\s)', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            logger.debug(f"Processing sentence: {sentence}")
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) <= base_chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
                logger.debug(f"Added to current_chunk: {current_chunk}")
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    logger.debug(f"Appended chunk: {current_chunk.strip()}")
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
            logger.debug(f"Final chunk: {current_chunk.strip()}")
        # Combine last two chunks if possible
        if len(chunks) >= 2:
            logger.debug(
                f"Checking combine: Last chunk len={len(chunks[-1])}, Second last len={len(chunks[-2])}")
            if len(chunks[-1]) + len(chunks[-2]) <= base_chunk_size:
                combined = f"{chunks[-2]} {chunks[-1]}"
                chunks[-2] = combined
                chunks.pop()
                logger.debug(f"Combined chunks: {combined}")
        logger.debug(f"Final chunks: {chunks}")
        return chunks

    def query_expansion(self, query: str) -> str:
        """Generate an expanded query for better retrieval (simplified HyDE)."""
        hypothetical_answer = f"Details about {query.lower()} in the context of job experience or skills."
        return f"{query} {hypothetical_answer}"

    def search(self, query: str, top_k: int = 5, hybrid_weight: float = 0.5) -> List[SearchResult]:
        """Perform hybrid vector and keyword search."""
        logger.debug(f"Search query: {query}, top_k: {top_k}")
        expanded_query = self.query_expansion(query)
        logger.debug(f"Expanded query: {expanded_query}")
        query_embedding = generate_embeddings(
            expanded_query, self.model, return_format="numpy")
        logger.debug(f"Query embedding shape: {query_embedding.shape}")
        query_embedding = query_embedding[0] if query_embedding.ndim > 1 else query_embedding
        vector_scores = []
        for i, chunk in enumerate(self.chunks):
            embedding = self.embeddings[i]
            embedding = embedding[0] if embedding.ndim > 1 else embedding
            score = float(np.dot(embedding, query_embedding) /
                          (np.linalg.norm(embedding) * np.linalg.norm(query_embedding)))
            vector_scores.append({"chunk": chunk, "score": score})
            logger.debug(f"Chunk {chunk['id']} score: {score}")

        tokenized_query = expanded_query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores = bm25_scores / \
            np.max(bm25_scores) if np.max(bm25_scores) > 0 else bm25_scores
        logger.debug(f"BM25 scores: {bm25_scores}")

        combined_scores = []
        for i, vector_score in enumerate(vector_scores):
            combined_score = hybrid_weight * \
                vector_score["score"] + (1 - hybrid_weight) * bm25_scores[i]
            combined_scores.append(
                {"chunk": vector_score["chunk"], "score": combined_score})
            logger.debug(
                f"Combined score for {vector_score['chunk']['id']}: {combined_score}")

        combined_scores.sort(key=lambda x: x["score"], reverse=True)
        logger.debug(
            f"Sorted results: {[x['chunk']['id'] for x in combined_scores[:top_k]]}")
        return combined_scores[:top_k]

    def evaluate_retrieval(self, query: str, relevant_chunk_ids: List[str], top_k: int = 5) -> float:
        """Evaluate retrieval quality using NDCG."""
        logger.debug(
            f"Evaluating retrieval for query: {query}, relevant IDs: {relevant_chunk_ids}")
        results = self.search(query, top_k)
        y_true = [[1 if result["chunk"]["id"]
                   in relevant_chunk_ids else 0 for result in results]]
        y_score = [[result["score"] for result in results]]
        logger.debug(f"y_true: {y_true}, y_score: {y_score}")
        if len(y_true[0]) < 2:
            y_true[0].extend([0] * (2 - len(y_true[0])))
            y_score[0].extend([0.0] * (2 - len(y_score[0])))
            logger.debug(f"Padded y_true: {y_true}, y_score: {y_score}")
        ndcg = ndcg_score(y_true, y_score)
        logger.debug(f"NDCG score: {ndcg}")
        return ndcg


class MetadataExtractor(Treeprocessor):
    """Markdown treeprocessor to extract metadata from specific sections."""

    def __init__(self, md, metadata_fields: Dict[str, str]):
        super().__init__(md)
        self.metadata_fields = metadata_fields
        self.metadata = {}
        self.in_personal_info = False

    def run(self, root: ET.Element) -> ET.Element:
        """Extract metadata from markdown AST based on configured fields."""
        logger.debug("Starting metadata extraction from markdown AST")
        for elem in root.iter():
            if elem.tag in ('h2', 'h3') and elem.text == "Personal Information":
                logger.debug("Found 'Personal Information' section")
                self.in_personal_info = True
                continue
            if self.in_personal_info and elem.tag == 'p' and elem.text:
                logger.debug(f"Processing paragraph: {elem.text[:100]}...")
                for field, pattern in self.metadata_fields.items():
                    if pattern.lower() in elem.text.lower():
                        self.metadata[field] = elem.text.strip()
                        logger.debug(
                            f"Extracted metadata: {field} = {elem.text.strip()}")
            if elem.tag in ('h2', 'h3') and elem.text != "Personal Information":
                self.in_personal_info = False  # Exit section when another header is found
        logger.debug(f"Extracted metadata: {self.metadata}")
        return root


class MetadataExtension(Extension):
    """Markdown extension to integrate MetadataExtractor."""

    def __init__(self, metadata_fields: Dict[str, str]):
        super().__init__()
        self.metadata_fields = metadata_fields

    def extendMarkdown(self, md):
        md.treeprocessors.register(MetadataExtractor(
            md, self.metadata_fields), 'metadata', 15)


def load_resume_markdown(file_path: str, candidate_id: str = "C123", metadata_fields: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Load and parse resume markdown file into a document dictionary with flexible metadata extraction."""
    try:
        if metadata_fields is None:
            metadata_fields = {
                "name": "Full Name",
                "location": "Location",
                "email": "Email"
            }

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Parse markdown to AST and extract metadata
        md = markdown.Markdown(
            extensions=[TocExtension(), MetadataExtension(metadata_fields)])
        html = md.convert(content)  # Triggers metadata extraction
        metadata = getattr(md, 'metadata', {})
        metadata.update({
            "candidate_id": candidate_id,
            "document_type": "resume"
        })

        # Clean content: remove table of contents and notes
        text = content
        toc_start = text.find("## Table of Contents")
        if toc_start != -1:
            toc_end = text.find("##", toc_start + 1)
            if toc_end == -1:
                toc_end = len(text)
            text = text[:toc_start] + text[toc_end:]
        notes_start = text.find("## Notes")
        if notes_start != -1:
            text = text[:notes_start]

        return {
            "id": candidate_id,
            "text": text.strip(),
            "metadata": metadata
        }
    except FileNotFoundError:
        logger.error(f"File {file_path} not found")
        raise
    except Exception as e:
        logger.error(f"Error loading markdown: {e}")
        raise
