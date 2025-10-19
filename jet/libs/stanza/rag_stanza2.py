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


# ---------------------------------------------------------------------------------------
# Initialize Stanza pipeline (English, fast mode)
# ---------------------------------------------------------------------------------------
def build_stanza_pipeline() -> stanza.Pipeline:
    """Build and return a lightweight Stanza pipeline."""
    stanza.download("en", processors="tokenize,pos,lemma,depparse,ner", verbose=False)
    return stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse,ner", use_gpu=False, verbose=False)


# ---------------------------------------------------------------------------------------
# Sentence parsing utility
# ---------------------------------------------------------------------------------------
def parse_sentences(text: str, nlp: stanza.Pipeline) -> List[Dict[str, Any]]:
    """Parse text into structured sentence objects."""
    doc = nlp(text)
    parsed = []

    for sent in doc.sentences:
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
    Combine parsed sentences into syntax-aware chunks.
    - Merges sentences until max_tokens is reached.
    - Computes a naive salience score based on entity density.
    """
    chunks: List[Dict[str, Any]] = []
    current_chunk: List[Dict[str, Any]] = []
    current_len = 0

    for i, sent in enumerate(parsed_sentences):
        sent_len = len(sent["tokens"])
        if current_len + sent_len > max_tokens and current_chunk:
            chunks.append(_finalize_chunk(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(sent)
        current_len += sent_len

    if current_chunk:
        chunks.append(_finalize_chunk(current_chunk))

    return chunks


def _finalize_chunk(sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Helper to finalize chunk structure and compute salience."""
    text = " ".join(s["text"] for s in sentences)
    all_entities = [ent.split(":")[0] for s in sentences for ent in s["entities"]]
    sent_indices = list(range(len(sentences)))

    salience = round(len(all_entities) * 1.2 + len(text) / 100, 2)  # simple heuristic

    return {
        "text": text,
        "sent_indices": sent_indices,
        "tokens": sum(len(s["tokens"]) for s in sentences),
        "entities": list(set(all_entities)),
        "salience": salience,
    }


# ---------------------------------------------------------------------------------------
# Demo entrypoint — used by tests and notebooks
# ---------------------------------------------------------------------------------------
def run_rag_stanza_demo(text: str) -> Dict[str, Any]:
    """
    Run the full Stanza-based RAG preprocessing pipeline.
    Returns a dict with sentence-level and chunk-level information.
    """
    print("=== Building Stanza pipeline (this may take a few seconds) ===")
    nlp = build_stanza_pipeline()

    print("\n=== Parsing sentences ===")
    parsed_sentences = parse_sentences(text, nlp)
    print(f"Total sentences parsed: {len(parsed_sentences)}")

    print("\n=== Creating context chunks for RAG ===")
    chunks = build_context_chunks(parsed_sentences, max_tokens=80)
    print(f"Generated {len(chunks)} chunks.\n")

    for i, ch in enumerate(chunks, 1):
        print(f">>> Chunk {i}")
        print(f"Sentences: {ch['sent_indices']}")
        print(f"Token count (approx): {ch['tokens']}")
        print(f"Salience score: {ch['salience']}")
        print(f"Text preview: {ch['text'][:180]}...\n")
        print(f"Entities in chunk: {', '.join(ch['entities']) if ch['entities'] else 'None'}\n")

    return {"parsed_sentences": parsed_sentences, "chunks": chunks}


# ---------------------------------------------------------------------------------------
# Allow running directly for manual demo
# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "OpenAI announced the GPT-5 model on October 15, 2025, marking a major leap "
        "in multimodal reasoning and multilingual understanding. "
        "The company claims that GPT-5 can handle text, image, and structured data "
        "simultaneously, offering developers a unified API for advanced RAG systems. "
        "Meanwhile, universities like Stanford and MIT are exploring Stanza-based "
        "syntactic chunking for retrieval-augmented generation improvements."
    )

    output = run_rag_stanza_demo(sample_text)
    print("\n=== Summary ===")
    print(f"Sentences parsed: {len(output['parsed_sentences'])}")
    print(f"Chunks created: {len(output['chunks'])}")
