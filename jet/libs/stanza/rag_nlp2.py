# rag_pipeline.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, TypedDict
import math

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Optional import for Stanza (if available)
try:
    import stanza
    _STANZA_AVAILABLE = True
except Exception:
    stanza = None  # type: ignore
    _STANZA_AVAILABLE = False


class ChunkMeta(TypedDict):
    doc_id: str
    chunk_id: str
    text: str
    source: Optional[str]


class RetrievalScore(TypedDict):
    similarity: float
    diversity: Optional[float]


class RetrievedChunk(TypedDict):
    meta: ChunkMeta
    score: RetrievalScore
    ner: Optional[List[Dict[str, str]]]
    topic: Optional[str]


def init_stanza_pipeline(lang: str = "en", processors: str = "tokenize,lemma,pos,ner") -> Optional[object]:
    """
    Initialize a stanza pipeline. Returns pipeline or None if stanza is unavailable.
    """
    if not _STANZA_AVAILABLE:
        return None
    # ensure models are downloaded externally prior to running in CI or runtime
    try:
        stanza.download(lang, quiet=True)
    except Exception:
        # ignore if already present or offline — calling code should handle None
        pass
    return stanza.Pipeline(lang=lang, processors=processors, use_gpu=False, verbose=False)


def preprocess_markdown_to_sentences(markdown_text: str, stanza_pipe: Optional[object] = None) -> List[str]:
    """
    Convert markdown to plain text minimally (strip markdown markers) and use stanza
    for sentence segmentation. Keep code blocks as single spans.
    NOTE: this function purposefully keeps logic simple and deterministic for tests.
    """
    # lightweight markdown strip:
    text = markdown_text
    # Remove fenced code blocks entirely or keep them as single chunks (here: keep)
    # collapse repeated newlines
    text = text.replace("\r\n", "\n")
    # remove basic markdown headers/emphasis/links inline for cleaner embeddings
    import re
    text = re.sub(r"```.*?```", lambda m: m.group(0).replace("\n", " "), text, flags=re.S)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # links -> anchor text
    text = re.sub(r"(^|\s)[#>*\-]{1,3}\s+", " ", text)  # headers/list bullets
    text = re.sub(r"`([^`]+)`", r"\1", text)  # inline code ticks
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\n{2,}", "\n\n", text)  # limit blank lines

    # If stanza is available, use its sentence splitter; otherwise, naive split by newlines and punctuation.
    if stanza_pipe:
        doc = stanza_pipe(text)
        sentences: List[str] = []
        for sent in doc.sentences:
            s = sent.text.strip()
            if s:
                sentences.append(s)
        return sentences
    else:
        # fallback: split on paragraph/newline then on sentence punctuation
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        sents = []
        for p in paras:
            # split by punctuation followed by space + capital letter or line break
            parts = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', p) if s.strip()]
            sents.extend(parts)
        return sents


def chunk_sentences(sentences: List[str], max_chars: int = 800) -> List[ChunkMeta]:
    """
    Combine contiguous sentences into chunks of at most `max_chars` characters.
    Returns ChunkMeta list with doc_id='doc' (caller may override).
    """
    chunks: List[ChunkMeta] = []
    cur: List[str] = []
    cur_len = 0
    chunk_idx = 0
    for i, s in enumerate(sentences):
        add_len = len(s) + (1 if cur else 0)
        if cur_len + add_len > max_chars and cur:
            chunks.append({"doc_id": "doc", "chunk_id": f"chunk-{chunk_idx}", "text": " ".join(cur), "source": None})
            chunk_idx += 1
            cur = [s]
            cur_len = len(s)
        else:
            cur.append(s)
            cur_len += add_len
    if cur:
        chunks.append({"doc_id": "doc", "chunk_id": f"chunk-{chunk_idx}", "text": " ".join(cur), "source": None})
    return chunks


def embed_texts(texts: List[str], model: Optional[SentenceTransformer] = None, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed texts using a provided SentenceTransformer model or by loading model_name.
    Returns (n_texts, dim) float32 embedding matrix.
    """
    if model is None:
        model = SentenceTransformer(model_name)
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype(np.float32)


def mmr_rerank(query_emb: np.ndarray, doc_embs: np.ndarray, top_k: int = 5, lambda_param: float = 0.7) -> Tuple[List[int], List[float]]:
    """
    Maximal Marginal Relevance ranking:
    - query_emb: (dim,) vector (normalized)
    - doc_embs: (n_docs, dim) matrix (normalized)
    Returns (ranked_indices, diversity_scores) where diversity_scores is per selected item (0..1).
    """
    # compute similarities
    sims = cosine_similarity(query_emb.reshape(1, -1), doc_embs).flatten()  # (n,)
    n = doc_embs.shape[0]
    selected: List[int] = []
    diversity_scores: List[float] = []

    if n == 0:
        return selected, diversity_scores

    # choose highest sim as first
    idx = int(np.argmax(sims))
    selected.append(idx)
    diversity_scores.append(0.0)  # first has no diversity baseline

    # precompute pairwise sims between docs
    pairwise = cosine_similarity(doc_embs)  # (n, n)

    while len(selected) < min(top_k, n):
        scores = []
        for i in range(n):
            if i in selected:
                scores.append(-math.inf)
                continue
            relevance = sims[i]
            # find max similarity to any selected
            max_sim_to_selected = max(pairwise[i, j] for j in selected) if selected else 0.0
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            scores.append(mmr_score)
        next_idx = int(np.argmax(scores))
        # diversity score defined as 1 - max_sim_to_selected (how different it is from selected set)
        max_sim = max(pairwise[next_idx, j] for j in selected) if selected else 0.0
        diversity_scores.append(float(1.0 - max_sim))
        selected.append(next_idx)

    return selected, diversity_scores


def retrieve(
    query: str,
    chunks: List[ChunkMeta],
    embedding_model: Optional[SentenceTransformer] = None,
    mmr: bool = False,
    mmr_top_k: int = 5,
    mmr_lambda: float = 0.7,
    stanza_pipe: Optional[object] = None,
) -> List[RetrievedChunk]:
    """
    Retrieve top chunks given a query. If mmr is True, apply MMR to diversify results.
    Returns RetrievedChunk list with similarity and optional diversity scores.
    """
    # 1) embed chunks and query
    texts = [c["text"] for c in chunks]
    chunk_embs = embed_texts(texts, model=embedding_model)
    query_emb = embed_texts([query], model=embedding_model)[0]

    # 2) compute similarity scores
    sims = cosine_similarity(query_emb.reshape(1, -1), chunk_embs).flatten().tolist()

    # 3) if MMR requested, rerank
    if mmr:
        selected_idxs, diversity_scores = mmr_rerank(query_emb, chunk_embs, top_k=mmr_top_k, lambda_param=mmr_lambda)
    else:
        # simple top-k by similarity
        order = np.argsort(-np.array(sims))[:mmr_top_k]
        selected_idxs = [int(i) for i in order]
        diversity_scores = [None] * len(selected_idxs)

    results: List[RetrievedChunk] = []
    # 4) extract NER if stanza provided
    for rank_pos, idx in enumerate(selected_idxs):
        meta = chunks[idx]
        sim_score = float(sims[idx])
        div_score = float(diversity_scores[rank_pos]) if mmr and diversity_scores[rank_pos] is not None else None
        ner = None
        if stanza_pipe:
            try:
                doc = stanza_pipe(meta["text"])
                ner = [{"text": e.text, "type": e.type} for sent in doc.sentences for e in sent.ents] if doc.sentences else []
            except Exception:
                ner = None
        results.append(RetrievedChunk(
            meta=meta,
            score=RetrievalScore(similarity=sim_score, diversity=div_score),
            ner=ner,
            topic=None
        ))
    return results
