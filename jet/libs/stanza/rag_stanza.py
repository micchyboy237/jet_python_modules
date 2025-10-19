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
from typing import Dict, Any, List
from tqdm import tqdm

from jet.libs.stanza.pipeline import StanzaPipelineCache


# ---------------------------------------------------------------------------------------
# Initialize Stanza pipeline (English, fast mode)
# ---------------------------------------------------------------------------------------
def build_stanza_pipeline() -> stanza.Pipeline:
    """Build and return a lightweight Stanza pipeline from cache."""
    cache = StanzaPipelineCache()
    return cache.get_pipeline(lang="en", processors="tokenize,pos,lemma,depparse,ner", use_gpu=True)


# ---------------------------------------------------------------------------------------
# Sentence parsing utility
# ---------------------------------------------------------------------------------------
def parse_sentences(text: str, nlp: stanza.Pipeline) -> List[Dict[str, Any]]:
    """Parse text into structured sentence objects with progress tracking."""
    doc = nlp(text)
    parsed = []
    for sent in tqdm(doc.sentences, desc="Parsing sentences"):
        tokens = [word.text for word in sent.words]
        pos_tags = [word.xpos for word in sent.words]
        lemmas = [word.lemma for word in sent.words]
        deps = [word.deprel for word in sent.words]
        entities = [f"{ent.text}:{ent.type}" for ent in sent.ents]
        parsed.append(
            {
                "text": sent.text.strip(),
                "tokens": tokens,
                "pos": pos_tags,
                "lemmas": lemmas,
                "entities": entities,
                "deps": deps,
            }
        )
    return parsed


# ---------------------------------------------------------------------------------------
# Context chunking for RAG
# ---------------------------------------------------------------------------------------
def build_context_chunks(parsed_sentences: List[Dict[str, Any]], max_tokens: int = 80) -> List[Dict[str, Any]]:
    """
    Combine parsed sentences into syntax-aware chunks with progress tracking.
    - Merges sentences until max_tokens is reached.
    - Computes a naive salience score based on entity density.
    """
    chunks: List[Dict[str, Any]] = []
    current_chunk: List[Dict[str, Any]] = []
    current_len = 0
    current_start_idx = 0
    for i, sent in enumerate(tqdm(parsed_sentences, desc="Building chunks")):
        sent_len = len(sent["tokens"])
        if current_len + sent_len > max_tokens and current_chunk:
            chunks.append(_finalize_chunk(current_chunk, start_idx=current_start_idx))
            current_chunk = [sent]
            current_len = sent_len
            current_start_idx = i
        else:
            current_chunk.append(sent)
            current_len += sent_len
    if current_chunk:
        chunks.append(_finalize_chunk(current_chunk, start_idx=current_start_idx))
    return chunks


def _finalize_chunk(sentences: List[Dict[str, Any]], start_idx: int) -> Dict[str, Any]:
    """Helper to finalize chunk structure and compute salience."""
    text = " ".join(s["text"] for s in sentences)
    all_entities = [ent.split(":")[0] for s in sentences for ent in s["entities"]]
    sent_indices = list(range(start_idx, start_idx + len(sentences)))
    salience = round(len(set(all_entities)) * 1.2 + len(text) / 100, 2)
    return {
        "text": text,
        "sent_indices": sent_indices,
        "tokens": sum(len(s["tokens"]) for s in sentences),
        "entities": list(set(all_entities)),
        "salience": salience,
    }
