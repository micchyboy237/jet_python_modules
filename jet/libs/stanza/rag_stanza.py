"""
Example module: Demonstrate how to use Stanza NLP for syntax-aware RAG context building.

This demo:
- Parses sentences using the Stanza NLP pipeline.
- Extracts tokens, POS tags, lemmas, and named entities.
- Groups sentences into syntax-aware context chunks.
- Computes a simple salience score per chunk (based on entity density and length).
- Returns structured results ready for downstream retrieval or embedding.
"""

from __future__ import annotations
import stanza
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from jet.libs.stanza.pipeline import StanzaPipelineCache

# ---------------------------------------------------------------------------------------
# Initialize Stanza pipeline (English, fast mode)
# ---------------------------------------------------------------------------------------
def build_stanza_pipeline(
    lang: str = "en",
    processors: str = "tokenize,pos,lemma,depparse,ner",
    use_gpu: bool = False
):
    """
    Create or retrieve a cached Stanza pipeline.
    Ensures compatibility with StanzaPipelineCache.get_pipeline signature.
    """
    cache = StanzaPipelineCache()
    return cache.get_pipeline(lang=lang, processors=processors, use_gpu=use_gpu)

# ---------------------------------------------------------------------------------------
# Sentence parsing utility
# ---------------------------------------------------------------------------------------
def parse_sentences(text: str, nlp: stanza.Pipeline) -> List[Dict[str, Any]]:
    """
    Parse text into structured sentence objects.
    Enhancements & fixes:
      - normalize start_char/end_char offsets (handle int, tuple, list, str)
      - tokens include offsets where available
      - entities emitted as dicts with normalized offsets and safe lemma lookup
      - defensive about missing/atypical stanza attribute shapes
    """
    def _normalize_offset(val: Any) -> Optional[int]:
        """Return an integer start offset where possible, else None."""
        if val is None:
            return None
        if isinstance(val, int):
            return int(val)
        if isinstance(val, (tuple, list)) and len(val) > 0:
            # Some stanza builds expose offsets as (start, end) or similar
            try:
                return int(val[0])
            except Exception:
                return None
        if isinstance(val, str) and val.isdigit():
            return int(val)
        # fallback: try to coerce
        try:
            return int(val)
        except Exception:
            return None

    def _entity_first_token_index(ent: Any) -> Optional[int]:
        """
        Safely obtain the 0-based token index corresponding to the first token of an entity.
        Handles cases where ent.tokens have token objects with 'id' possibly being tuple or int.
        """
        try:
            if hasattr(ent, "tokens") and ent.tokens:
                tok = ent.tokens[0]
                # token id may be int, tuple, or str like "1"
                tid = getattr(tok, "id", None)
                if isinstance(tid, int):
                    return tid - 1 if tid > 0 else 0
                if isinstance(tid, (tuple, list)) and len(tid) > 0:
                    try:
                        return int(tid[0]) - 1
                    except Exception:
                        return None
                if isinstance(tid, str) and tid.isdigit():
                    return int(tid) - 1
        except Exception:
            return None
        return None

    doc = nlp(text)
    parsed: List[Dict[str, Any]] = []
    for si, sent in enumerate(tqdm(doc.sentences, desc="Parsing sentences")):
        # normalize sentence offsets defensively
        s_start = _normalize_offset(getattr(sent, "start_char", None))
        s_end = _normalize_offset(getattr(sent, "end_char", None))

        # build token list with offsets and lemma/pos
        tokens = []
        for w in sent.words:
            tok_start = _normalize_offset(getattr(w, "start_char", None))
            tok_end = _normalize_offset(getattr(w, "end_char", None))
            token_obj = {
                "text": w.text,
                "lemma": getattr(w, "lemma", w.text),
                "pos": getattr(w, "xpos", getattr(w, "upos", None)),
                "deprel": getattr(w, "deprel", None),
                "start_char": tok_start,
                "end_char": tok_end,
                "id": getattr(w, "id", None),
            }
            tokens.append(token_obj)

        # map entities to structured dicts (defensive handling)
        entities: List[Dict[str, Any]] = []
        ents = getattr(sent, "ents", None)
        if ents is None:
            # fallback: use doc-level entities and filter by normalized offsets
            doc_ents = getattr(doc, "entities", []) or []
            filtered = []
            for e in doc_ents:
                e_start = _normalize_offset(getattr(e, "start_char", None))
                e_end = _normalize_offset(getattr(e, "end_char", None))
                if e_start is None or e_end is None:
                    continue
                # ensure sentence offsets exist; if not, include based on containment in text length
                if s_start is None or s_end is None:
                    if 0 <= e_start <= len(text):
                        filtered.append(e)
                else:
                    if e_start >= s_start and e_end <= s_end:
                        filtered.append(e)
            ents = filtered

        for ent in ents or []:
            ent_start = _normalize_offset(getattr(ent, "start_char", None))
            ent_end = _normalize_offset(getattr(ent, "end_char", None))
            # safe lemma: try to map to first token in tokens list
            lemma_val = None
            try:
                token_idx = _entity_first_token_index(ent)
                if token_idx is not None and 0 <= token_idx < len(tokens):
                    lemma_val = tokens[token_idx].get("lemma")
            except Exception:
                lemma_val = None

            entities.append({
                "text": getattr(ent, "text", None),
                "type": getattr(ent, "type", None),
                "start_char": ent_start,
                "end_char": ent_end,
                "lemma": lemma_val or getattr(ent, "text", None)
            })

        # root/head token (use dependency info to identify important sentence anchor)
        head_token_text = None
        for w in sent.words:
            if getattr(w, "head", 0) == 0:
                head_token_text = w.text
                break

        parsed.append(
            {
                "sentence_id": si,
                "text": sent.text.strip(),
                "start_char": s_start,
                "end_char": s_end,
                "tokens": tokens,
                "token_count": len(tokens),
                "head_token": head_token_text,
                "entities": entities,
                "deps": [getattr(w, "deprel", None) for w in sent.words],
            }
        )
    return parsed

# ---------------------------------------------------------------------------------------
# Context chunking for RAG
# ---------------------------------------------------------------------------------------
def build_context_chunks(parsed_sentences: List[Dict[str, Any]], max_tokens: int = 80, preserve_entities: bool = True) -> List[Dict[str, Any]]:
    """
    Combine parsed sentences into syntax-aware chunks.
    - Preserves sentence boundaries.
    - Optionally avoids splitting inside entities (preserve_entities=True).
    - Uses the new _finalize_chunk_v2 which includes char offsets and metadata.
    """
    chunks: List[Dict[str, Any]] = []
    current_chunk: List[Dict[str, Any]] = []
    current_len = 0
    current_start_idx = 0
    for i, sent in enumerate(tqdm(parsed_sentences, desc="Building chunks")):
        sent_len = sent["token_count"]
        # if we would split inside an entity, prefer to include the sentence (simple heuristic)
        if current_len + sent_len > max_tokens and current_chunk:
            chunks.append(_finalize_chunk_v2(current_chunk, start_idx=current_start_idx))
            current_chunk = [sent]
            current_len = sent_len
            current_start_idx = i
        else:
            current_chunk.append(sent)
            current_len += sent_len
    if current_chunk:
        chunks.append(_finalize_chunk_v2(current_chunk, start_idx=current_start_idx))
    return chunks

def _finalize_chunk_v2(sentences: List[Dict[str, Any]], start_idx: int) -> Dict[str, Any]:
    """
    Create richer chunk structure:
      - start_char/end_char (from first/last sentence if available)
      - entity list as dicts with counts
      - token counts, sentence indices, and existing salience formula
      - record stanza version & note for reproducibility (if available)
    """
    text = " ".join(s["text"] for s in sentences)
    # char offsets defensive: prefer first and last sentence offsets, else None
    start_char = next((s.get("start_char") for s in sentences if s.get("start_char") is not None), None)
    end_char = next((s.get("end_char") for s in reversed(sentences) if s.get("end_char") is not None), None)

    all_entities = [ent["text"] for s in sentences for ent in s["entities"]]
    unique_entities = list(dict.fromkeys(all_entities))  # preserve order, unique

    sent_indices = [s["sentence_id"] for s in sentences]
    token_count = sum(s["token_count"] for s in sentences)

    # enhanced salience: combine unique entity count, head_token presence and length
    head_tokens = [s.get("head_token") for s in sentences if s.get("head_token")]
    head_bonus = len([h for h in head_tokens if h]) * 0.3
    salience = round(len(unique_entities) * 1.4 + token_count / 120 + head_bonus, 3)

    chunk_meta = {
        "text": text,
        "sent_indices": sent_indices,
        "tokens": token_count,
        "entities": unique_entities,
        "entity_count": len(all_entities),
        "salience": salience,
        "start_char": start_char,
        "end_char": end_char,
        # record stanza version for reproducibility
        "stanza_version": getattr(stanza, "__version__", None),
    }
    return chunk_meta

def rerank_chunks_for_query(chunks: List[Dict[str, Any]], query: str, nlp: stanza.Pipeline, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Query-aware reranking:
      1. Parse the query with the same pipeline.
      2. Extract query entity lemmas and keywords.
      3. Score chunks by mixing previous salience and overlap features.
      4. Return top_k sorted chunks (highest first), with added 'query_score' and 'final_score'.
    Notes: This is a lightweight, deterministic re-ranker. For stronger results mix in vector similarity.
    """
    q_parsed = parse_sentences(query, nlp)
    # flatten query lemmas & entity lemmas
    q_lemmas = set()
    q_entity_texts = set()
    for s in q_parsed:
        for t in s["tokens"]:
            if t.get("lemma"):
                q_lemmas.add(t["lemma"].lower())
        for ent in s["entities"]:
            q_entity_texts.add(ent["lemma"].lower() if ent.get("lemma") else ent["text"].lower())

    scored: List[Dict[str, Any]] = []
    for c in chunks:
        # entity overlap
        chunk_entity_lower = {e.lower() for e in c.get("entities", [])}
        entity_overlap = len(chunk_entity_lower.intersection(q_entity_texts))

        # lemma overlap (approx): token-level naive match using chunk text split
        chunk_text_lemmas = set([w.lower() for w in c.get("text", "").split() if len(w) > 2])
        lemma_overlap = len(chunk_text_lemmas.intersection(q_lemmas))

        # final mixing strategy: preserve original salience but boost with overlaps
        base_salience = float(c.get("salience", 0))
        query_score = entity_overlap * 2.0 + lemma_overlap * 0.5
        final_score = round(base_salience * 0.6 + query_score * 1.2, 3)

        new_c = dict(c)
        new_c.update({"query_score": query_score, "entity_overlap": entity_overlap, "lemma_overlap": lemma_overlap, "final_score": final_score})
        scored.append(new_c)

    scored_sorted = sorted(scored, key=lambda x: x["final_score"], reverse=True)
    return scored_sorted[:top_k]
