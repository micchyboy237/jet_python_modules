# /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/adapters/llama_cpp/examples/chunk_strategies/02_demo_fixed_size_chunking.py
"""Demo: TokenAwareFixedSizeChunker for code/structured content.

This input is a Python function with no meaningful sentence boundaries.
Sentence-aware chunking would produce degenerate results here since the
entire block is one "sentence" to NLTK. The fixed-size strategy slides
a token-exact window across the raw token stream, producing uniform chunks
that preserve syntactic structure through consistent boundary placement
and efficient batch decoding.
"""

from jet.adapters.llama_cpp.chunk_strategy_utils import get_chunker

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


def main() -> None:
    chunker = get_chunker("fixed", model="qwen3.5:2b")

    chunks = chunker.chunk(
        text=CODE_TEXT,
        chunk_size=80,
        chunk_overlap=16,
        min_chunk_size=20,
        buffer=4,
    )

    print(f"Strategy: TokenAwareFixedSizeChunker")
    print(f"Input length: {len(CODE_TEXT)} chars")
    print(f"Output chunks: {len(chunks)}")
    print("-" * 60)

    for i, chunk in enumerate(chunks):
        token_count = len(chunker.size_fn(chunk))
        print(f"\n[Chunk {i}] ({token_count} tokens)")
        print(chunk)

    # Verify uniformity: all non-tail chunks should be near-effective size
    effective = 80 - 4
    sizes = [len(chunker.size_fn(c)) for c in chunks]
    print("\n" + "-" * 60)
    print(f"Chunk token sizes: {sizes}")
    print(f"Effective budget: {effective} tokens")
    non_tail = sizes[:-1] if len(sizes) > 1 else sizes
    if non_tail and all(s >= effective - 2 for s in non_tail):
        print("Non-tail chunks are uniformly packed near budget.")
    else:
        print("Note: Some non-tail chunks are below expected fill level.")


if __name__ == "__main__":
    main()
