"""Demo: using LlamacppEmbeddings standalone and with a Chonkie chunker.

Prereqs:
  - A llama.cpp server running with an embedding model loaded
    (LLAMA_CPP_EMBED_URL / LLAMA_CPP_EMBED_MODEL env vars, or defaults
    in jet.adapters.llama_cpp.config).
"""

from jet.adapters.chonkie.llamacpp_embeddings import LlamacppEmbeddings
from jet.logger import logger

# 1. Basic single / batch embedding -----------------------------------------
embeddings = LlamacppEmbeddings()  # uses config.EMBED_MODEL by default

query = "What is a giant panda?"
docs = [
    "The giant panda is a bear species endemic to China.",
    "Python is a high-level programming language.",
    "Pandas eat bamboo and live in mountainous regions.",
]

query_vec = embeddings.embed_query(query)  # uses EMBED_QUERY_PREFIX
doc_vecs = embeddings.embed_batch(docs)  # uses EMBED_DOC_PREFIX

logger.info(f"Query vector shape: {query_vec.shape}")
logger.info(f"Doc vectors: {len(doc_vecs)} x {doc_vecs[0].shape}")
logger.info(f"Embedding dimension (lazy-resolved): {embeddings.dimension}")

# 2. Similarity ranking (uses BaseEmbeddings.similarity, cosine by default) --
scored = [
    (doc, embeddings.similarity(query_vec, vec)) for doc, vec in zip(docs, doc_vecs)
]
scored.sort(key=lambda x: x[1], reverse=True)

logger.info("Ranked docs by similarity to query:")
for doc, score in scored:
    logger.info(f"  {score:.4f}  {doc}")

# 3. Tokenizer access ---------------------------------------------------------
tokenizer = embeddings.get_tokenizer()
logger.info(f"Tokenizer: {tokenizer}")

# 4. Plug straight into a Chonkie chunker ------------------------------------
from chonkie import SemanticChunker

chunker = SemanticChunker(embedding_model=embeddings, chunk_size=256)

text = (
    "The giant panda is a bear species endemic to China. "
    "It primarily eats bamboo and lives in mountainous regions. "
    "In contrast, Python is a high-level, general-purpose programming language. "
    "It emphasizes code readability and supports multiple programming paradigms."
)
chunks = chunker(text)

logger.info(f"Chunked text into {len(chunks)} semantic chunks:")
for i, chunk in enumerate(chunks):
    logger.info(f"  [{i}] tokens={chunk.token_count} text={chunk.text[:60]!r}...")

print(repr(embeddings))
