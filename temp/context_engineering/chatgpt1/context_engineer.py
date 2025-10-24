# context_engineer.py
from typing import List, Tuple, TypedDict, Optional

import numpy as np

from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.adapters.llama_cpp.llm import LlamacppLLM

# ---- Types ----
class RetrievedDoc(TypedDict):
    id: int
    text: str
    score: float  # similarity score (higher == more similar)

# ---- Utilities ----
def chunk_text(text: str, max_chars: int = 2000) -> List[str]:
    """
    Naive chunking by characters that tries to preserve sentence boundaries.
    Returns list of chunks <= max_chars.
    """
    if not text:
        return []
    parts: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        # try to extend to next newline or sentence end if possible (look ahead)
        if end < n:
            sep_pos = text.rfind("\n", start, end)
            if sep_pos <= start:
                sep_pos = text.rfind(". ", start, end)
            if sep_pos > start:
                end = sep_pos + 1
        parts.append(text[start:end].strip())
        start = end
    return parts

# ---- Embedding & Retrieval ----
class Embedder:
    """Wrapper for LlamacppEmbedding – returns L2-normalized numpy arrays."""
    def __init__(
        self,
        model_name: str = "embeddinggemma",
        base_url: str = "http://shawn-pc.local:8081/v1",
    ):
        self.client = LlamacppEmbedding(model=model_name, base_url=base_url, use_cache=True)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)   # 384 = default embedding size for embeddinggemma
        embs = self.client.get_embeddings(texts, return_format="numpy")
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embs / norms

def cosine_similarities(query_emb: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
    """Return cosine similarities between query_emb (d,) and corpus_embs (n, d)."""
    if corpus_embs.size == 0:
        return np.array([], dtype=float)
    # both should be L2-normalized already
    return (corpus_embs @ query_emb).astype(float)

def retrieve_top_k(
    query: str,
    documents: List[str],
    embedder: Embedder,
    k: int = 3
) -> List[RetrievedDoc]:
    """
    Embeds query and document chunks, computes cosine similarity, and returns top-k docs.
    Documents are provided as raw texts; chunking is applied per-doc.
    """
    # 1) chunk each document and keep mapping
    flat_chunks: List[str] = []
    chunk_to_doc: List[int] = []
    for i, doc in enumerate(documents):
        for chunk in chunk_text(doc, max_chars=1200):
            flat_chunks.append(chunk)
            chunk_to_doc.append(i)

    if not flat_chunks:
        return []

    # 2) embed
    corpus_embs = embedder.embed_texts(flat_chunks)  # (n, d)
    query_emb = embedder.embed_texts([query])[0]     # (d,)

    # 3) compute sims and pick top chunks
    sims = cosine_similarities(query_emb, corpus_embs)  # (n,)
    top_idx = list(np.argsort(sims)[::-1][:k])

    results: List[RetrievedDoc] = []
    seen_docs = set()
    for idx in top_idx:
        doc_index = chunk_to_doc[idx]
        seen_docs.add(doc_index)
        results.append(RetrievedDoc(id=doc_index, text=flat_chunks[idx], score=float(sims[idx])))

    return results

# ---- Condense / Context Management ----
def condense_context_via_llm(prompt_prefix: str, contexts: List[str], model: str = "qwen3-instruct-2507:4b") -> str:
    llm = LlamacppLLM(model=model, base_url="http://shawn-pc.local:8080/v1", verbose=True)
    instruction = (
        prompt_prefix
        + "\n\n"
        + "Combine the following document excerpts into a concise, factual summary (<= 300 words). "
        + "Preserve important facts and attributes. Do not hallucinate.\n\n"
    )
    for i, c in enumerate(contexts, start=1):
        instruction += f"--- Document {i} ---\n{c}\n"
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": instruction},
    ]
    return llm.chat(messages, temperature=0.0, max_tokens=400)

# ---- Prompt assembly and LLM generation ----
def build_prompt(query: str, retrieved: List[RetrievedDoc], condensed_context: Optional[str] = None) -> str:
    """
    Assemble the final prompt using template strategy: short system instruction, context, query,
    and expected output format.
    """
    system = (
        "You are an expert assistant. Use only the provided context to answer the user's query. "
        "If the context does not contain the answer, say 'I don't know'."
    )

    ctx_block = ""
    if condensed_context:
        ctx_block = f"Context summary:\n{condensed_context}\n\n"
    else:
        # assemble raw retrieved docs
        for r in retrieved:
            ctx_block += f"[doc#{r['id']} score={r['score']:.3f}]\n{r['text']}\n\n"

    prompt = (
        f"{system}\n\n"
        f"{ctx_block}"
        f"User question:\n{query}\n\n"
        "Instructions:\n"
        "1) Answer succinctly and cite doc ids where you used the facts (e.g., [doc#0]).\n"
        "2) If you are unsure or the context lacks the facts, say 'I don't know'.\n"
    )
    return prompt

def generate_answer(prompt: str, model: str = "qwen3-instruct-2507:4b") -> str:
    llm = LlamacppLLM(model=model, base_url="http://shawn-pc.local:8080/v1")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return llm.chat(messages, temperature=0.0, max_tokens=512)

# ---- High-level pipeline ----
def answer_with_context_engineering(
    query: str,
    documents: List[str],
    embedder: Optional[Embedder] = None,
    k: int = 4,
    condense: bool = True,
) -> Tuple[str, List[RetrievedDoc]]:
    """
    Full pipeline:
      1. retrieve top-k chunks
      2. optionally condense them using the LLM (to save tokens)
      3. assemble prompt and call the LLM for final answer
    Returns tuple (answer_text, retrieved_docs)
    """
    embedder = embedder or Embedder()
    retrieved = retrieve_top_k(query, documents, embedder, k=k)
    if not retrieved:
        prompt = build_prompt(query, retrieved, condensed_context=None)
        return generate_answer(prompt), retrieved

    if condense:
        # condense texts (safest to pass excerpts only)
        excerpts = [r["text"] for r in retrieved]
        condensed = condense_context_via_llm("Please condense these excerpts into a short factual summary.", excerpts)
        prompt = build_prompt(query, retrieved, condensed_context=condensed)
    else:
        prompt = build_prompt(query, retrieved, condensed_context=None)

    answer = generate_answer(prompt)
    return answer, retrieved

# ---- Example usage guard ----
if __name__ == "__main__":
    docs = [
        "Project Neptune launched in 2021 and its focus is on freshwater research. The project lead is Dr. Martinez.",
        "A short note: Neptune is the nickname for our internal freshwater monitoring system.",
        "Budget for Project Neptune was $2.4M for the 2022 fiscal year."
    ]
    q = "Who leads Project Neptune?"
    # ensure env var OPENAI_API_KEY set before running
    ans, r = answer_with_context_engineering(q, docs, condense=True)
    print("ANSWER:\n", ans)
    print("RETRIEVED:", r)