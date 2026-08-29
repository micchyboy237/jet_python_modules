# rag_module_v1/formatting.py

from jet.adapters.llama_cpp.token_utils import count_tokens

from .schemas import RetrievedChunk


def format_context(
    results: list[RetrievedChunk],
    max_tokens: int,
) -> tuple[str, bool]:
    parts: list[str] = []
    truncated = False

    for r in results:
        chunk = r.chunk

        block = (
            f"[Source: {chunk.doc_title} | {chunk.chunk_id}]\n{chunk.content.strip()}\n"
        )

        candidate = "\n\n".join(parts + [block])
        token_count = count_tokens(candidate)

        if token_count <= max_tokens:
            parts.append(block)
            continue

        truncated = True
        break

    return "\n\n".join(parts).strip(), truncated
