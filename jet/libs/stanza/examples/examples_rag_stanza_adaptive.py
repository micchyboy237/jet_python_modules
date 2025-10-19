# jet_python_modules/jet/libs/stanza/examples/examples_rag_stanza_adaptive.py
"""
examples_rag_stanza_adaptive.py
Usage examples for `rag_stanza_adaptive.py`.

Demonstrates how to:
  1. Parse text using Stanza and analyze syntax complexity.
  2. Adaptively chunk sentences based on dependency and entity density.
  3. Visualize dependency trees in Graphviz DOT format.
  4. Run the full adaptive RAG demo.

Run:
    python examples_rag_stanza_adaptive.py
"""

from pathlib import Path
from jet.libs.stanza.rag_stanza_adaptive import (
    adaptive_chunk_score,
    build_adaptive_chunks,
    visualize_sentence_dependency_dot,
    run_rag_stanza_adaptive_demo,
)
from jet.libs.stanza.rag_stanza import build_stanza_pipeline, parse_sentences

EXAMPLE_TEXT = (
    "OpenAI released GPT-5 in October 2025, marking a major leap in multimodal reasoning. "
    "The model can analyze text, images, and structured data simultaneously. "
    "Analysts believe this capability may transform enterprise retrieval systems. "
    "Meanwhile, research labs such as MIT CSAIL and ETH Zurich are developing adaptive RAG pipelines "
    "that combine Stanza parsing with dependency-based chunking."
)


def example_adaptive_score():
    """Example: compute adaptive complexity scores for parsed sentences."""
    print("=== Example: Adaptive Sentence Scoring ===")
    nlp = build_stanza_pipeline()
    parsed = parse_sentences(EXAMPLE_TEXT, nlp)
    for i, sent in enumerate(parsed[:3], 1):
        score = adaptive_chunk_score(sent)
        print(f"Sentence {i}: score={score:.3f}, tokens={len(sent['tokens'])}, entities={len(sent['entities'])}")


def example_adaptive_chunking():
    """Example: adaptive chunk construction."""
    print("\n=== Example: Adaptive Chunking ===")
    nlp = build_stanza_pipeline()
    parsed = parse_sentences(EXAMPLE_TEXT, nlp)
    chunks = build_adaptive_chunks(parsed, base_max_tokens=60)
    print(f"Generated {len(chunks)} adaptive chunks.")
    for i, c in enumerate(chunks, 1):
        print(f"\n>>> Chunk {i}")
        print(f"Sent indices: {c['sent_indices']}")
        print(f"Tokens: {c['tokens']}")
        print(f"Salience: {c['salience']}")
        print(f"Entities: {', '.join(c['entities']) if c['entities'] else 'None'}")
        print(f"Text preview: {c['text'][:150]}...")


def example_dependency_visualization():
    """Example: visualize dependency tree for the first parsed sentence."""
    print("\n=== Example: Dependency Visualization ===")
    nlp = build_stanza_pipeline()
    parsed = parse_sentences(EXAMPLE_TEXT, nlp)
    doc = nlp(EXAMPLE_TEXT)
    # Attach heads for visualization
    for sent_obj, stanza_sent in zip(parsed, doc.sentences):
        sent_obj["heads"] = [w.head for w in stanza_sent.words]

    out_path = Path("output/deptree_example.dot")
    visualize_sentence_dependency_dot(parsed[0], out_path)
    print(f"Dependency tree DOT written to: {out_path.resolve()}")


def example_full_adaptive_demo():
    """Example: run the full adaptive RAG demo."""
    print("\n=== Example: Full Adaptive RAG Demo ===")
    results = run_rag_stanza_adaptive_demo(EXAMPLE_TEXT)
    print(f"\nParsed {len(results['parsed_sentences'])} sentences.")
    print(f"Generated {len(results['chunks'])} chunks.")
    print(f"DOT file: {results['dot_file']}")


if __name__ == "__main__":
    example_adaptive_score()
    example_adaptive_chunking()
    example_dependency_visualization()
    example_full_adaptive_demo()
