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
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
import stanza

from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.libs.bertopic.examples.mock import EMBED_MODEL
from jet.wordnet.text_chunker import ChunkResult, ChunkResultMeta, ChunkResultWithMeta, chunk_texts_with_data


# =============== MODELS ======================

def get_stanza_pipeline(lang: str = "en"):
    """Load and cache Stanza pipeline for sentence validation."""
    return stanza.Pipeline(lang=lang, processors="tokenize,pos,lemma,depparse", use_gpu=False, verbose=False)


def get_embedder(model_name: str = "embeddinggemma"):
    # """Load a pretrained SentenceTransformer embedder."""
    # return SentenceTransformer(model_name)
    return LlamacppEmbedding(model="embeddinggemma")


# =============== DATA STRUCTURES ==============

@dataclass
class Chunk:
    id: str
    text: str
    section_title: Optional[str] = None
    header_level: Optional[int] = None
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
    chunk_size: int = 200,
    chunk_overlap: int = 32,
    model: str = EMBED_MODEL,
    stride_ratio: float = 0.3,
    embedder: Optional[SentenceTransformer] = None,
    ignore_links: bool = False,
    truncate: bool = False,
) -> List[Chunk]:
    """Chunk markdown by headers."""
    headers = derive_by_header_hierarchy(markdown_text, ignore_links=True)
    header_md_contents = [f"{header['parent_header'] or ''}\n{header['header']}\n{header['content']}".strip() for header in headers]

    texts = header_md_contents
    doc_ids = [header["id"] for header in headers]

    # Map header id to header metadata (assuming doc_ids are unique and order-aligned)
    header_id_to_meta: dict[str, ChunkResultMeta] = {}
    for header in headers:
        header_meta: ChunkResultMeta = {
            "doc_id": header["id"],
            "doc_index": header["doc_index"],
            "header": header["header"],
            "level": header.get("level"),
            "parent_header": header.get("parent_header"),
            "parent_level": header.get("parent_level"),
            "source": header.get("source"),
            "tokens": header.get("tokens"),
        }
        header_id_to_meta[header["id"]] = header_meta

    base_chunks: List[ChunkResult] = chunk_texts_with_data(
        texts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model=model,
        ids=doc_ids,
    )

    # Attach section meta-data from the corresponding HeaderDoc to each chunk
    enriched_chunks: List[ChunkResultWithMeta] = []
    for chunk in base_chunks:
        doc_id = chunk["doc_id"]
        if doc_id in header_id_to_meta:
            chunk_meta = header_id_to_meta[doc_id]
        else:
            # Safely fill with correct types for fields
            chunk_meta: ChunkResultMeta = {
                "doc_id": doc_id,
                "doc_index": -1,
                "header": "",
                "level": None,
                "parent_header": None,
                "parent_level": None,
                "source": None,
                "tokens": [],
            }
        chunk_with_meta: ChunkResultWithMeta = {**chunk, "meta": chunk_meta}
        enriched_chunks.append(chunk_with_meta)

    if truncate:
        enriched_chunks = [chunk for chunk in enriched_chunks if chunk.get("chunk_index", 0) == 0]

    return [Chunk(
        id=chunk["id"],
        text=chunk["content"],
        section_title=chunk["meta"]["header"].lstrip('#').strip(),
        header_level=chunk["meta"]["level"],
        metadata={"section": chunk["meta"]["header"].lstrip('#').strip(), "level": chunk["meta"]["level"]},
    ) for chunk in enriched_chunks]


# =============== RETRIEVAL =====================

def mmr_select_with_scores(
    query_emb,
    doc_embs,
    top_k: int = 5,
    diversity: float = 0.5
):
    """
    MMR selection that also returns similarity and diversity scores.
    """
    if isinstance(query_emb, torch.Tensor):
        query_emb = query_emb.detach().to("cpu")
    if isinstance(doc_embs, list):
        doc_embs = [e.detach().to("cpu") if isinstance(e, torch.Tensor) else torch.tensor(e) for e in doc_embs]
        doc_embs = torch.stack(doc_embs)

    sim_query = util.cos_sim(query_emb, doc_embs)[0].cpu().numpy()
    sim_doc = util.cos_sim(doc_embs, doc_embs).cpu().numpy()

    selected = [int(np.argmax(sim_query))]
    scores = [{"idx": selected[0], "similarity": float(sim_query[selected[0]]), "diversity": 0.0}]

    candidates = list(range(len(doc_embs)))
    for _ in range(1, top_k):
        mmr_scores = []
        for idx in candidates:
            if idx in [s["idx"] for s in scores]:
                continue
            relevance = sim_query[idx]
            diversity_score = max(sim_doc[idx][[s["idx"] for s in scores]]) if scores else 0
            score = (1 - diversity) * relevance - diversity * diversity_score
            mmr_scores.append((idx, relevance, diversity_score, score))
        if not mmr_scores:
            break
        best = max(mmr_scores, key=lambda x: x[3])
        scores.append({"idx": best[0], "similarity": float(best[1]), "diversity": float(best[2])})

    return scores


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
            return chunk_markdown_sections(text)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return chunk_by_sentences(sentences, stride_ratio=self.stride_ratio, embedder=self.embedder)

    def retrieve(self, query: str, chunks: List[Chunk], top_k: int = 5, diversity: Optional[float] = None):
        """
        Retrieve top chunks using cosine similarity or MMR if diversity is provided.
        Adds similarity/diversity scores to each chunk's metadata.
        """
        all_emb = self.embedder([query] + [c.text for c in chunks])
        query_emb = all_emb[0]
        doc_embs = all_emb[1:]

        sim_scores = util.cos_sim(query_emb, doc_embs)[0].cpu().numpy()

        if diversity is not None:
            selected_info = mmr_select_with_scores(query_emb, doc_embs, top_k=top_k, diversity=diversity)
        else:
            top_idx = np.argsort(sim_scores)[::-1][:top_k]
            selected_info = [{"idx": int(i), "similarity": float(sim_scores[i]), "diversity": 0.0} for i in top_idx]

        results = []
        for s in selected_info:
            c = chunks[s["idx"]]
            c.metadata = c.metadata or {}
            c.metadata["similarity"] = s["similarity"]
            if diversity is not None:
                c.metadata["diversity"] = s["diversity"]
            results.append(c)
        return results
