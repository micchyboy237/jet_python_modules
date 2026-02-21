import logging
import os
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.logging import RichHandler
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

# Rich logging (beautiful console output)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("hybrid_rrf")
console = Console()


@dataclass
class Document:
    """Generic, reusable document container (type-safe for all components)."""

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LocalEmbedder:
    """Thin wrapper around sentence-transformers – generic & reusable."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


class DocumentProcessor:
    """Unstructured loading + chunking. Two tiny focused methods."""

    def load_directory(self, input_dir: str) -> List[Any]:
        """Partition every local file with Unstructured (auto strategy)."""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Directory not found: {input_dir}")
        console.print(
            f"[bold blue]Loading files from {input_dir} with Unstructured...[/]"
        )
        all_elements: List[Any] = []
        files = [f for f in input_path.rglob("*.*") if f.is_file()]
        for file_path in tqdm(files, desc="Partitioning files"):
            try:
                elements = partition(str(file_path), strategy="auto")
                for el in elements:
                    if hasattr(el, "metadata"):
                        el.metadata["source"] = str(file_path)
                all_elements.extend(elements)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Skipped {file_path}: {e}")
        logger.info(f"Extracted {len(all_elements)} raw elements")
        return all_elements

    def chunk_elements(self, elements: List[Any]) -> List[Document]:
        """Semantic chunking with chunk_by_title (best practice for RAG)."""
        console.print("[bold blue]Chunking by title...[/]")
        chunks = chunk_by_title(elements)
        docs: List[Document] = []
        for chunk in chunks:
            meta = (
                chunk.metadata.to_dict()
                if hasattr(chunk.metadata, "to_dict")
                else dict(chunk.metadata or {})
            )
            meta["chunk_id"] = str(uuid.uuid4())  # stable key for RRF dedup
            docs.append(Document(page_content=chunk.text, metadata=meta))
        logger.info(f"Created {len(docs)} chunks")
        return docs


class VectorStore:
    """Persistent Chroma vector index – zero-config."""

    def __init__(self, persist_dir: str = "./chroma_index"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="local_docs")

    def add_documents(self, docs: List[Document], embedder: LocalEmbedder) -> None:
        if not docs:
            return
        console.print("[bold blue]Building Chroma vector index...[/]")
        texts = [d.page_content for d in docs]
        embeddings = embedder.embed_documents(texts)
        ids = [d.metadata["chunk_id"] for d in docs]
        metadatas = [d.metadata for d in docs]
        self.collection.add(
            documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas
        )

    def similarity_search(
        self, query: str, k: int, embedder: LocalEmbedder
    ) -> List[Document]:
        query_emb = embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            include=["documents", "metadatas"],
        )
        docs: List[Document] = []
        # Defensive: handle empty result structure
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])
        num_results = len(results.get("ids", [[]])[0]) if results.get("ids") else 0
        for i in range(num_results):
            pc = documents[0][i] if len(documents) > 0 and len(documents[0]) > i else ""
            meta = (
                metadatas[0][i] if len(metadatas) > 0 and len(metadatas[0]) > i else {}
            )
            docs.append(
                Document(
                    page_content=pc,
                    metadata=meta,
                )
            )
        return docs


class BM25Store:
    """Pure BM25 keyword index."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.docs: List[Document] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def add_documents(self, docs: List[Document]) -> None:
        if not docs:
            return
        console.print("[bold blue]Building BM25 keyword index...[/]")
        self.docs = list(docs)
        corpus = [self._tokenize(d.page_content) for d in docs]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 20) -> List[Document]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(self._tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :k
        ]
        return [self.docs[i] for i in top_indices]


def reciprocal_rank_fusion(
    results_lists: List[List["Document"]],
    weights: Optional[List[float]] = None,
    rrf_k: int = 60,
) -> List["Document"]:
    """Standard RRF – tiny, generic, no external libs."""
    if weights is None:
        weights = [1.0] * len(results_lists)
    score_dict: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}
    for res_list, weight in zip(results_lists, weights):
        for rank, doc in enumerate(res_list):
            d_id = doc.metadata.get("chunk_id") or str(id(doc))
            doc_map[d_id] = doc
            score_dict[d_id] += weight / (rank + rrf_k)
    # sort by fused score
    sorted_ids = sorted(
        score_dict.keys(), key=lambda d_id: score_dict[d_id], reverse=True
    )
    return [doc_map[d_id] for d_id in sorted_ids]


class HybridRetriever:
    """Combines vector + BM25 with RRF. Easy to extend."""

    def __init__(
        self, vector_store: VectorStore, bm25_store: BM25Store, embedder: LocalEmbedder
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        vector_results = self.vector_store.similarity_search(
            query, k=20, embedder=self.embedder
        )
        bm25_results = self.bm25_store.search(query, k=20)
        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results], weights=[0.7, 0.3]
        )
        return fused[:k]


class HybridRRFPipeline:
    """Complete pipeline – ingest once, query forever."""

    def __init__(
        self,
        input_dir: str,
        persist_dir: str = "./chroma_index",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.input_dir = input_dir
        self.persist_dir = persist_dir
        self.processor = DocumentProcessor()
        self.embedder = LocalEmbedder(embedding_model)
        self.vector_store = VectorStore(persist_dir)
        self.bm25_store = BM25Store()
        self.retriever: Optional[HybridRetriever] = None

    def ingest(self) -> None:
        console.rule("[bold green]Hybrid RRF Pipeline – Ingestion[/]")
        elements = self.processor.load_directory(self.input_dir)
        docs = self.processor.chunk_elements(elements)
        self.vector_store.add_documents(docs, self.embedder)
        self.bm25_store.add_documents(docs)
        self.retriever = HybridRetriever(
            self.vector_store, self.bm25_store, self.embedder
        )
        console.print(
            "[bold green]✅ Hybrid RRF ready (Unstructured + Chroma + BM25 + RRF)[/]"
        )

    def query(self, query: str, k: int = 5) -> List[Document]:
        if not self.retriever:
            raise ValueError("Call ingest() first")
        console.print(f"[bold]Query:[/] {query}")
        return self.retriever.retrieve(query, k)


if __name__ == "__main__":
    # Point to your local data folder
    pipeline = HybridRRFPipeline(input_dir="./data")
    pipeline.ingest()
    results = pipeline.query(
        "How does Reciprocal Rank Fusion improve hybrid search on local documents?"
    )
    for i, doc in enumerate(results, 1):
        console.print(
            f"\n[bold cyan]Result {i} (source: {doc.metadata.get('source')})[/]"
        )
        if len(doc.page_content) > 500:
            console.print(doc.page_content[:500] + "...")
        else:
            console.print(doc.page_content)
