# jet_python_modules/jet/adapters/llama_cpp/chunk_strategies/examples/02_demo_fixed_size_chunking.py
"""Demo: TokenAwareFixedSizeChunker with overlap verification.

Demonstrates token-level sliding window chunking on Python code where
sentence boundaries are meaningless. Verifies that consecutive chunks
share exactly the configured number of overlap tokens through direct
token ID comparison.
"""

from jet.adapters.llama_cpp.chunk_strategies import detect_token_overlap, get_chunker

CODE_TEXT = """\
def compute_attention_scores(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attn_output, attn_weights = compute_attention_scores(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.dropout(self.w_o(attn_output))
"""

CHUNK_SIZE = 80
CHUNK_OVERLAP = 16
MIN_CHUNK_SIZE = 20
BUFFER = 4


def main() -> None:
    chunker = get_chunker("fixed", model="qwen3.5:2b")

    chunks = chunker.chunk(
        text=CODE_TEXT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
        buffer=BUFFER,
    )

    effective = CHUNK_SIZE - BUFFER
    step = CHUNK_SIZE - CHUNK_OVERLAP - BUFFER

    print(f"Strategy: TokenAwareFixedSizeChunker")
    print(
        f"Config: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, "
        f"min={MIN_CHUNK_SIZE}, buffer={BUFFER} → effective={effective}, step={step}"
    )
    print(f"Input length: {len(CODE_TEXT)} chars")
    print(f"Output chunks: {len(chunks)}")
    print("=" * 60)

    # Pre-tokenize all chunks for overlap verification
    chunk_token_lists = [chunker.size_fn(c) for c in chunks]

    for i, chunk in enumerate(chunks):
        tok_count = len(chunk_token_lists[i])
        print(f"\n[Chunk {i}] ({tok_count} tokens)")
        print(chunk)

        # Verify overlap via direct token ID comparison
        if i < len(chunks) - 1:
            actual_overlap = detect_token_overlap(
                chunk_token_lists[i], chunk_token_lists[i + 1]
            )
            status = "✅" if actual_overlap == CHUNK_OVERLAP else "⚠️"
            print(
                f"  ↕ Overlap with Chunk {i + 1}: {actual_overlap} tokens "
                f"(expected {CHUNK_OVERLAP}) {status}"
            )

    # Summary statistics
    sizes = [len(t) for t in chunk_token_lists]
    print("\n" + "=" * 60)
    print(f"Chunk token sizes: {sizes}")
    print(f"Effective budget: {effective} tokens")
    non_tail = sizes[:-1] if len(sizes) > 1 else sizes
    if non_tail and all(s >= effective - 2 for s in non_tail):
        print("✅ Non-tail chunks are uniformly packed near budget.")
    else:
        print("⚠️  Note: Some non-tail chunks are below expected fill level.")


if __name__ == "__main__":
    main()
