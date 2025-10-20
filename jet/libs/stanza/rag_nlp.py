"""
Reusable RAG NLP pipeline using Stanza + BERTopic + SentenceTransformers.

Supports:
  - Markdown-aware section chunking
  - Sliding-window sentence chunking
  - MMR-based retrieval
  - Optional BERTopic topic tagging
"""

from __future__ import annotations
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
import stanza


# =============== MODELS ======================

def get_stanza_pipeline(lang: str = "en"):
    """Load and cache Stanza pipeline for sentence validation."""
    return stanza.Pipeline(lang=lang, processors="tokenize,pos,lemma,depparse", use_gpu=False, verbose=False)


def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Load a pretrained SentenceTransformer embedder."""
    return SentenceTransformer(model_name)


# =============== DATA STRUCTURES ==============

@dataclass
class Chunk:
    id: str
    text: str
    section_title: Optional[str] = None
    header_level: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


# =============== UTILITIES ====================

HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def is_valid_sentence(sentence: str, nlp=None) -> bool:
    """Check if a sentence is grammatically valid using Stanza."""
    if not sentence or len(sentence.strip()) < 3:
        return False
    nlp = nlp or get_stanza_pipeline()
    doc = nlp(sentence)
    if not doc.sentences:
        return False
    s = doc.sentences[0]
    # Must contain at least one subject and one predicate
    pos_tags = [w.upos for w in s.words]
    return "NOUN" in pos_tags or "PRON" in pos_tags and "VERB" in pos_tags


def split_by_markdown_headers(markdown_text: str) -> List[Dict[str, str]]:
    """Split markdown into sections using headers."""
    matches = list(HEADER_PATTERN.finditer(markdown_text))
    if not matches:
        return [{"header": None, "level": 0, "content": markdown_text.strip()}]

    sections = []
    for i, match in enumerate(matches):
        header_level = len(match.group(1))
        header_text = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        content = markdown_text[start:end].strip()
        sections.append({
            "header": header_text,
            "level": header_level,
            "content": content
        })
    return sections


# =============== CHUNKING =====================

def chunk_by_sentences(
    sentences: List[str],
    max_tokens: int = 200,
    stride_ratio: float = 0.3,
    embedder: Optional[SentenceTransformer] = None
) -> List[Chunk]:
    """Merge sentences into overlapping chunks using sliding window."""
    chunks: List[Chunk] = []
    idx = 0
    stride_tokens = int(max_tokens * (1 - stride_ratio))
    sent_lens = [len(s.split()) for s in sentences]
    i = 0

    while i < len(sentences):
        buf = []
        buf_words = 0
        for j in range(i, len(sentences)):
            wlen = sent_lens[j]
            if buf_words + wlen > max_tokens:
                break
            buf.append(sentences[j])
            buf_words += wlen
        if not buf:
            i += 1
            continue

        text = " ".join(buf).strip()
        chunks.append(Chunk(id=f"chunk_{idx}", text=text))
        idx += 1

        step_words = 0
        step_sentences = 0
        for s_len in sent_lens[i:]:
            if step_words >= stride_tokens:
                break
            step_words += s_len
            step_sentences += 1
        i += max(1, step_sentences)

    # Compute embeddings
    if embedder:
        embs = embedder.encode([c.text for c in chunks], convert_to_tensor=True)
        for c, e in zip(chunks, embs):
            c.embedding = e

    return chunks


def chunk_markdown_sections(
    markdown_text: str,
    max_tokens: int = 200,
    stride_ratio: float = 0.3,
    embedder: Optional[SentenceTransformer] = None
) -> List[Chunk]:
    """Chunk markdown by headers, then apply sliding-window chunking within sections."""
    sections = split_by_markdown_headers(markdown_text)
    all_chunks: List[Chunk] = []
    idx = 0

    for sec in sections:
        sentences = re.split(r'(?<=[.!?])\s+', sec["content"])
        section_chunks = chunk_by_sentences(
            sentences, max_tokens=max_tokens, stride_ratio=stride_ratio, embedder=embedder
        )
        for c in section_chunks:
            c.section_title = sec["header"]
            c.header_level = sec["level"]
            c.id = f"chunk_{idx}"
            c.metadata = {"section": sec["header"], "level": sec["level"]}
            all_chunks.append(c)
            idx += 1

    return all_chunks


# =============== RETRIEVAL =====================

def mmr_select(
    query_emb,
    doc_embs,
    top_k: int = 5,
    diversity: float = 0.5
):
    """
    Maximal Marginal Relevance (MMR) for diverse retrieval.
    """
    import torch

    # Ensure tensors are on CPU and detached
    if isinstance(query_emb, torch.Tensor):
        query_emb = query_emb.detach().to("cpu")

    # Convert document embeddings to tensor
    if isinstance(doc_embs, list):
        doc_embs = [e.detach().to("cpu") if isinstance(e, torch.Tensor) else torch.tensor(e) for e in doc_embs]
        doc_embs = torch.stack(doc_embs)

    # Compute cosine similarities safely on CPU
    sim_query = util.cos_sim(query_emb, doc_embs)[0].cpu().numpy()
    sim_doc = util.cos_sim(doc_embs, doc_embs).cpu().numpy()

    selected = [int(np.argmax(sim_query))]
    candidates = list(range(len(doc_embs)))

    for _ in range(1, top_k):
        mmr_scores = []
        for idx in candidates:
            if idx in selected:
                continue
            relevance = sim_query[idx]
            diversity_score = max(sim_doc[idx][selected]) if selected else 0
            score = (1 - diversity) * relevance - diversity * diversity_score
            mmr_scores.append((idx, score))
        if not mmr_scores:
            break
        best = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best)

    return selected


# =============== TOPIC TAGGING =================

def tag_topics(chunks: List[Chunk], embedder: SentenceTransformer) -> BERTopic:
    """Fit BERTopic model on chunks to derive topic labels."""
    topic_model = BERTopic()
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, convert_to_tensor=False)
    topics, _ = topic_model.fit_transform(texts, embeddings)
    for c, t in zip(chunks, topics):
        c.metadata = c.metadata or {}
        c.metadata["topic"] = int(t)
    return topic_model


# =============== MAIN PIPELINE =================

class RAGPipeline:
    """Reusable RAG NLP pipeline."""

    def __init__(self, model_name="all-MiniLM-L6-v2", use_markdown=True, stride_ratio=0.3):
        self.embedder = get_embedder(model_name)
        self.use_markdown = use_markdown
        self.stride_ratio = stride_ratio

    def prepare_chunks(self, text: str) -> List[Chunk]:
        if self.use_markdown:
            return chunk_markdown_sections(text, stride_ratio=self.stride_ratio, embedder=self.embedder)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return chunk_by_sentences(sentences, stride_ratio=self.stride_ratio, embedder=self.embedder)

    def retrieve(self, query: str, chunks: List[Chunk], top_k: int = 5, diversity: float = 0.5):
        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        doc_embs = [c.embedding for c in chunks if c.embedding is not None]
        selected_idx = mmr_select(query_emb, doc_embs, top_k=top_k, diversity=diversity)
        return [chunks[i] for i in selected_idx]

