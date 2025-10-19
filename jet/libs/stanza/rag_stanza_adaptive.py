"""
Adaptive Stanza-based RAG preprocessing with dependency visualization.
Enhancements:
- Adaptive chunking based on syntax complexity (sentence length + entity density)
- Dependency tree visualization using DOT format
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List

from jet.libs.stanza.rag_stanza import build_stanza_pipeline, parse_sentences
from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name


def adaptive_chunk_score(sentence: Dict[str, Any]) -> float:
    """
    Compute adaptive score to decide how 'complex' a sentence is.
    Combines entity count, dependency length, and token count.
    """
    entities = len(sentence.get("entities", []))
    tokens = len(sentence.get("tokens", []))
    deps = len(sentence.get("deps", []))
    return round((entities * 1.5) + (tokens / 15) + (deps / 20), 3)


def build_adaptive_chunks(
    parsed_sentences: List[Dict[str, Any]],
    base_max_tokens: int = 80
) -> List[Dict[str, Any]]:
    """
    Dynamically merges sentences into chunks based on syntax complexity.
    More complex sentences reduce the effective chunk size.
    """
    chunks: List[Dict[str, Any]] = []
    current_chunk: List[Dict[str, Any]] = []
    current_len = 0
    dynamic_limit = base_max_tokens

    for i, sent in enumerate(parsed_sentences):
        sent_len = len(sent["tokens"])
        score = adaptive_chunk_score(sent)
        # adjust limit dynamically (reduce size for complex sentences)
        dynamic_limit = max(40, base_max_tokens - int(score * 3))
        if current_len + sent_len > dynamic_limit and current_chunk:
            chunks.append(_finalize_adaptive_chunk(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(sent)
        current_len += sent_len
    if current_chunk:
        chunks.append(_finalize_adaptive_chunk(current_chunk))
    return chunks


def _finalize_adaptive_chunk(sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute salience and merge adaptive chunk sentences."""
    text = " ".join(s["text"] for s in sentences)
    entities = [ent.split(":")[0] for s in sentences for ent in s.get("entities", [])]
    salience = round(len(entities) * 1.3 + len(text) / 90, 2)
    return {
        "text": text,
        "sent_indices": list(range(len(sentences))),
        "tokens": sum(len(s["tokens"]) for s in sentences),
        "entities": list(set(entities)),
        "salience": salience,
    }


def visualize_sentence_dependency_dot(parsed_sent: Dict[str, Any], out_path: str) -> Path:
    """
    Writes Graphviz DOT representation of the dependency tree for the parsed sentence.
    Each token is a node; edges are labeled with dependency relations.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tokens = parsed_sent.get("tokens", [])
    heads = parsed_sent.get("heads", [])
    deps = parsed_sent.get("deps", [])

    lines = ["digraph G {", "  node [shape=plaintext];"]
    for idx, tok in enumerate(tokens, start=1):
        lines.append(f'  n{idx} [label="{idx}: {tok}"];')
    for idx, head in enumerate(heads, start=1):
        if head == 0:
            lines.append(f'  root -> n{idx} [label="root"];')
        elif head <= len(tokens):
            rel = deps[idx - 1] if idx - 1 < len(deps) else ""
            lines.append(f'  n{head} -> n{idx} [label="{rel}"];')
    lines.append("}")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def run_rag_stanza_adaptive_demo(text: str) -> Dict[str, Any]:
    """
    Full adaptive RAG demo with dependency visualization output.
    """
    print("=== Building Stanza adaptive pipeline ===")
    nlp = build_stanza_pipeline()
    parsed_sentences = parse_sentences(text, nlp)

    # Add heads for visualization (ensure parse_sentences includes .head if not)
    for sent in nlp(text).sentences:
        heads = [w.head for w in sent.words]
        for parsed, head in zip(parsed_sentences, [heads]):
            parsed["heads"] = head

    print(f"Total sentences parsed: {len(parsed_sentences)}")

    chunks = build_adaptive_chunks(parsed_sentences)
    print(f"Generated {len(chunks)} adaptive chunks.\n")

    # Visualize the first sentence for debugging/demo
    dot_file = visualize_sentence_dependency_dot(parsed_sentences[0], f"{get_entry_file_dir()}/generated/{os.path.splitext(get_entry_file_name())[0]}/ldeptree_demo.dot")
    print(f"Dependency tree DOT file written to: {dot_file.resolve()}")
    return {"parsed_sentences": parsed_sentences, "chunks": chunks, "dot_file": str(dot_file)}
