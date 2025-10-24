# context_engineering.py
from __future__ import annotations
from typing import List, Dict, Any, TypedDict, Callable, Optional, Tuple, Iterable
from dataclasses import dataclass
import math
import re

from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.adapters.llama_cpp.llm import LlamacppLLM

# Optional imports (used only when plugging real backends)
try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # allow tests to run without faiss

# --- Types ------------------------------------------------
class Document(TypedDict):
    id: str
    text: str
    meta: Dict[str, Any]

class Chunk(TypedDict):
    id: str
    doc_id: str
    text: str
    meta: Dict[str, Any]

Embedding = List[float]
EmbeddingModel = Any  # abstract: must implement .encode(list[str]) -> List[List[float]]
LLMModel = Any  # abstract: must implement .generate(prompt: str, **kwargs) -> str

# --- Utilities ------------------------------------------------
def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# --- Chunking ------------------------------------------------
@dataclass
class ChunkConfig:
    max_tokens: int = 200  # approximate token-equivalent (heuristic)
    overlap: int = 50      # overlap token-equivalent
    split_on_sentence: bool = True

def _approx_token_count(text: str) -> int:
    # conservative heuristic: 1 token ≈ 4 chars
    return max(1, math.ceil(len(text) / 4))

def chunk_text(doc_id: str, text: str, cfg: ChunkConfig) -> List[Chunk]:
    """
    Break `text` into chunks, with overlap. This function is intentionally simple,
    uses sentence boundaries when enabled, and falls back to fixed-size splits.
    """
    text = _normalize_whitespace(text)
    if not text:
        return []
    tokens = _approx_token_count(text)
    # sentence-based splitting
    if cfg.split_on_sentence:
        # very lightweight sentence splitter
        sents = re.split(r'(?<=[.!?])\s+', text)
        chunks: List[str] = []
        current = ""
        current_tokens = 0
        for s in sents:
            s = s.strip()
            if not s:
                continue
            s_tokens = _approx_token_count(s)
            if current_tokens + s_tokens <= cfg.max_tokens:
                current = (current + " " + s).strip() if current else s
                current_tokens += s_tokens
            else:
                if current:
                    chunks.append(current)
                # if sentence itself too large, split it
                if s_tokens > cfg.max_tokens:
                    # fixed splits for this long sentence
                    parts = _fixed_split(s, cfg.max_tokens)
                    chunks.extend(parts[:-1])
                    current = parts[-1]
                    current_tokens = _approx_token_count(current)
                else:
                    current = s
                    current_tokens = s_tokens
        if current:
            chunks.append(current)
    else:
        chunks = _fixed_split(text, cfg.max_tokens)

    # apply overlap
    result: List[Chunk] = []
    chunk_id = 0
    for i, c in enumerate(chunks):
        # overlap: include previous suffix up to overlap tokens
        if cfg.overlap > 0 and i > 0:
            prev = chunks[i - 1]
            prev_tokens = _approx_token_count(prev)
            overlap_chars = int(cfg.overlap * 4)
            suffix = prev[-overlap_chars:] if overlap_chars < len(prev) else prev
            c = (suffix + " " + c).strip()
        result.append(Chunk(id=f"{doc_id}__chunk__{chunk_id}", doc_id=doc_id, text=_normalize_whitespace(c), meta={}))
        chunk_id += 1
    return result

def _fixed_split(text: str, max_tokens: int) -> List[str]:
    approx_chars = max(1, max_tokens * 4)
    parts = [text[i:i+approx_chars] for i in range(0, len(text), approx_chars)]
    return [p.strip() for p in parts if p.strip()]

# --- Vector Indexing (pluggable) -----------------------------------------
@dataclass
class InMemoryIndex:
    """
    Simple in-memory vector index for tests and small usage.
    Expects consumer to provide embeddings for items in the same order as `add`.
    """
    vectors: List[Embedding]
    chunks: List[Chunk]

    def __init__(self):
        self.vectors = []
        self.chunks = []

    def add(self, embs: List[Embedding], chunks: List[Chunk]) -> None:
        assert len(embs) == len(chunks)
        self.vectors.extend(embs)
        self.chunks.extend(chunks)

    def search(self, query_emb: Embedding, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        # simple cosine similarity
        def dot(a: Embedding, b: Embedding) -> float:
            return sum(x*y for x,y in zip(a,b))
        def norm(a: Embedding) -> float:
            return math.sqrt(sum(x*x for x in a)) or 1e-12
        qn = norm(query_emb)
        results: List[Tuple[Chunk, float]] = []
        for vec, chk in zip(self.vectors, self.chunks):
            score = dot(vec, query_emb) / (norm(vec) * qn)
            results.append((chk, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

# --- Pipeline ------------------------------------------------
@dataclass
class RetrievalConfig:
    chunk_cfg: ChunkConfig = ChunkConfig()
    top_k: int = 5
    context_token_budget: int = 1500  # approximate token budget for context included in prompt

class ContextEngine:
    def __init__(
        self,
        embedding_fn: Optional[Callable[[List[str]], List[Embedding]]] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        embedding_base_url: str = "http://shawn-pc.local:8081/v1",
        llm_base_url: str = "http://shawn-pc.local:8080/v1",
        embedding_model: str = "embeddinggemma",
        llm_model: str = "qwen3-instruct-2507:4b",
        index: Optional[InMemoryIndex] = None,
    ):
        self.embedding_client = LlamacppEmbedding(model=embedding_model, base_url=embedding_base_url, use_cache=True)
        self.llm_client = LlamacppLLM(model=llm_model, base_url=llm_base_url, verbose=True)

        def default_embed(texts: List[str]) -> List[Embedding]:
            arr = self.embedding_client.get_embeddings(texts, return_format="numpy")
            return arr.tolist()

        def default_llm(prompt: str) -> str:
            messages = [{"role": "user", "content": prompt}]
            return self.llm_client.chat(messages, temperature=0.0)

        self.embedding_fn = embedding_fn or default_embed
        self.llm_fn = llm_fn or default_llm
        self.index = index or InMemoryIndex()
        self._chunks_added = False

    def ingest_documents(self, documents: Iterable[Document], chunk_cfg: Optional[ChunkConfig] = None) -> None:
        cfg = chunk_cfg or ChunkConfig()
        chunks: List[Chunk] = []
        for doc in documents:
            chunks.extend(chunk_text(doc_id=doc["id"], text=doc["text"], cfg=cfg))
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        embs = self.embedding_fn(texts)
        self.index.add(embs, chunks)
        self._chunks_added = True

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Chunk, float]]:
        if top_k is None:
            top_k = 5
        q_emb = self.embedding_fn([query])[0]
        return self.index.search(q_emb, top_k=top_k)

    def _assemble_context(self, retrieved: List[Tuple[Chunk, float]], token_budget: int) -> str:
        # greedy pack by approximate token count until budget
        packed: List[str] = []
        used = 0
        for chk, score in retrieved:
            tkns = _approx_token_count(chk["text"])
            if used + tkns > token_budget:
                continue
            packed.append(f"[source_id={chk['doc_id']}|chunk_id={chk['id']}|score={score:.3f}]\n{chk['text']}")
            used += tkns
        return "\n\n".join(packed)

    def generate(self, query: str, cfg: Optional[RetrievalConfig] = None) -> str:
        cfg = cfg or RetrievalConfig()
        if not self._chunks_added:
            # ingest empty / no docs behavior: still call LLM with query-only prompt
            retrieved = []
        else:
            retrieved = self.retrieve(query, top_k=cfg.top_k)
        context = self._assemble_context(retrieved, token_budget=cfg.context_token_budget)
        prompt = self._build_prompt(query, context)
        return self.llm_fn(prompt)

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build a compact, instruction-style prompt. Keep this simple & auditable.
        """
        instructions = [
            "You are an assistant that answers user questions using ONLY the provided context.",
            "If the answer is not contained in the context, say 'I don't know.' and do not hallucinate.",
        ]
        prompt_parts = [
            "### CONTEXT",
            context if context else "(no context available)",
            "### INSTRUCTIONS",
            "\n".join(instructions),
            "### USER QUERY",
            query,
            "### ANSWER (concise, cite source_id if used)"
        ]
        return "\n\n".join(prompt_parts)
