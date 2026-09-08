"""Conversation-aware chunking for chat logs and support tickets.

Chunks multi-turn conversations by speaker turns while preserving speaker
identity and timestamps as metadata. Enables filtered retrieval like
"What did Alice say about pricing?" targeting specific participants.
"""

from __future__ import annotations

import numpy as np
from jet.adapters.chonkie.llamacpp_embeddings import LlamacppEmbeddings
from jet.adapters.chonkie.llamacpp_genie import LlamacppGenie
from jet.logger import logger

# ---------------------------------------------------------------------------
# Simulated conversation log
# ---------------------------------------------------------------------------

CONVERSATION = [
    {
        "speaker": "Alice",
        "ts": "2026-09-08T09:00:00",
        "text": "Hey team, we need to discuss the new pricing model for Q4.",
    },
    {
        "speaker": "Bob",
        "ts": "2026-09-08T09:01:00",
        "text": "Sure. I've been analyzing competitor pricing. Most charge per-seat now.",
    },
    {
        "speaker": "Alice",
        "ts": "2026-09-08T09:02:00",
        "text": "Per-seat won't work for our enterprise clients. They want flat-rate tiers.",
    },
    {
        "speaker": "Carol",
        "ts": "2026-09-08T09:03:00",
        "text": "What about usage-based? We could meter API calls and storage.",
    },
    {
        "speaker": "Bob",
        "ts": "2026-09-08T09:04:00",
        "text": "Usage-based is fair but hard to predict revenue. Maybe hybrid?",
    },
    {
        "speaker": "Alice",
        "ts": "2026-09-08T09:05:00",
        "text": "Hybrid makes sense. Base platform fee plus usage overage for API calls.",
    },
    {
        "speaker": "Carol",
        "ts": "2026-09-08T09:06:00",
        "text": "I'll draft a proposal with three tiers: Starter, Growth, Enterprise.",
    },
    {
        "speaker": "Alice",
        "ts": "2026-09-08T09:07:00",
        "text": "Great. Include annual discount options too. 20% off for yearly commitment.",
    },
    {
        "speaker": "Bob",
        "ts": "2026-09-08T09:08:00",
        "text": "Should we grandfather existing customers at current rates for 6 months?",
    },
    {
        "speaker": "Alice",
        "ts": "2026-09-08T09:09:00",
        "text": "Yes, 6-month grace period. Send migration emails 30 days before switch.",
    },
    {
        "speaker": "Carol",
        "ts": "2026-09-08T09:10:00",
        "text": "I'll also prepare FAQ docs for the sales team. When do we launch?",
    },
    {
        "speaker": "Alice",
        "ts": "2026-09-08T09:11:00",
        "text": "Target October 15th. Let's reconvene Friday to review Carol's draft.",
    },
]


# ---------------------------------------------------------------------------
# Conversation chunker
# ---------------------------------------------------------------------------


def chunk_conversation(
    turns: list[dict],
    max_turns_per_chunk: int = 4,
) -> list[dict]:
    """Group conversation turns into overlapping chunks preserving metadata.

    Each chunk contains up to max_turns_per_chunk consecutive turns.
    Metadata includes all speakers present, time range, and turn count.
    """
    chunks = []
    for i in range(0, len(turns), max_turns_per_chunk):
        window = turns[i : i + max_turns_per_chunk]
        combined_text = "\n".join(
            f"[{t['ts'][11:16]}] {t['speaker']}: {t['text']}" for t in window
        )
        speakers = list(dict.fromkeys(t["speaker"] for t in window))  # ordered unique
        chunks.append(
            {
                "text": combined_text,
                "speakers": speakers,
                "start_ts": window[0]["ts"],
                "end_ts": window[-1]["ts"],
                "turn_count": len(window),
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Metadata-filtered vector store
# ---------------------------------------------------------------------------


class ConversationStore:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.metadata: list[dict] = []
        self.vectors: np.ndarray | None = None

    def add(self, texts: list[str], vectors: np.ndarray, meta: list[dict]) -> None:
        self.texts.extend(texts)
        self.metadata.extend(meta)
        self.vectors = (
            vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        )

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 3,
        speaker: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        if self.vectors is None:
            return []
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        sims = np.dot(self.vectors, query_vec) / norms

        indices = list(range(len(self.texts)))
        if speaker:
            indices = [
                i for i in indices if speaker in self.metadata[i].get("speakers", [])
            ]

        filtered_sims = sims[indices]
        top_idx = np.argsort(filtered_sims)[::-1][:top_k]
        return [
            (self.texts[indices[i]], float(filtered_sims[i]), self.metadata[indices[i]])
            for i in top_idx
        ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    embeddings = LlamacppEmbeddings(batch_size=64, show_progress=True)
    genie = LlamacppGenie(temperature=0.3)
    store = ConversationStore()

    # 1. Chunk conversation
    logger.info("=== Chunking Conversation ===")
    chunks = chunk_conversation(CONVERSATION, max_turns_per_chunk=4)
    for i, c in enumerate(chunks, 1):
        logger.info(
            f"  Chunk {i}: speakers={c['speakers']}, turns={c['turn_count']}, "
            f"time={c['start_ts'][11:16]}-{c['end_ts'][11:16]}"
        )

    # 2. Embed and store
    texts = [c["text"] for c in chunks]
    vectors = np.array(embeddings.embed_batch(texts))
    store.add(texts, vectors, chunks)

    # 3. Global query
    query = "What was decided about pricing?"
    logger.info(f"=== Global Query: '{query}' ===")
    q_vec = embeddings.embed(query)
    results = store.search(q_vec, top_k=2)
    print("\n🌍 Global Results:")
    for text, score, meta in results:
        print(
            f"  Score: {score:.4f} | Speakers: {meta['speakers']} | Time: {meta['start_ts'][11:16]}-{meta['end_ts'][11:16]}"
        )
        print(f"  {text[:120]}...\n")

    # 4. Speaker-filtered query
    logger.info("=== Filtered Query: speaker='Alice', topic='pricing' ===")
    alice_results = store.search(q_vec, top_k=2, speaker="Alice")
    print("🎯 Alice's Turns Only:")
    for text, score, meta in alice_results:
        print(
            f"  Score: {score:.4f} | Time: {meta['start_ts'][11:16]}-{meta['end_ts'][11:16]}"
        )
        print(f"  {text[:120]}...\n")

    # 5. Generate attributed answer
    context = "\n\n".join(f"[{m['start_ts'][11:16]}] {t}" for t, _, m in alice_results)
    prompt = f"Using ONLY Alice's statements below, summarize her position on pricing:\n\n{context}\n\nSummary:"
    answer = genie.generate(prompt)
    print(f"💡 Alice's Position:\n{answer}")


if __name__ == "__main__":
    main()
