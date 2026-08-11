import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import asyncio
import hashlib
import json
import logging
import time
from typing import Any, TypedDict

import chromadb
import trafilatura
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from config import (
    BUDGETS,
    CACHE_DB,
    GRAMMAR_DIR,
    MAX_DEPTH,
    MAX_ITERATIONS,
    MAX_TOTAL_TOKENS,
    MAX_WALL_SECONDS,
    RERANK_TOP_K,
    SEARXNG_CATEGORIES,
    SEARXNG_ENGINES,
    SEARXNG_MAX_RESULTS,
    SEARXNG_MIN_SCORE,
    SEARXNG_QUERY_URL,
    SEARXNG_USE_CACHE,
    SEMANTIC_DEDUP_THRESHOLD,
    VECTOR_DB_PATH,
)
from jet.adapters.llama_cpp.chunking_utils import (
    chunk_texts_with_data,
    truncate_texts,
)
from jet.adapters.llama_cpp.config import EMBED_LG_MODEL, LLM_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.factory import get_llm_client
from jet.adapters.llama_cpp.rerank_utils import rerank
from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.search.searxng import search_searxng
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from playwright.async_api import async_playwright

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("webswarm")

DEFAULT_LLM_MODEL = LLM_MODEL
DEFAULT_EMBED_MODEL = EMBED_LG_MODEL

# Chunking configuration
CHUNK_SIZE = int(os.getenv("SWARM_CHUNK_SIZE", "256"))
CHUNK_OVERLAP = int(os.getenv("SWARM_CHUNK_OVERLAP", "50"))
MIN_CHUNK_SIZE = int(os.getenv("SWARM_MIN_CHUNK_SIZE", "64"))
ENABLE_CHUNKING = os.getenv("SWARM_ENABLE_CHUNKING", "true").lower() == "true"
MERGE_CHUNKS_ON_RECALL = (
    os.getenv("SWARM_MERGE_CHUNKS_ON_RECALL", "true").lower() == "true"
)


class JetEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB-compatible wrapper for jet.adapters.llama_cpp.embed."""

    def __call__(self, input: Documents) -> Embeddings:
        return embed(
            input,
            model=DEFAULT_EMBED_MODEL,
            return_format="list",
            show_progress=True,
        )

    def name(self) -> str:
        return "jet_llama_cpp_embed"


class LocalLLMClient:
    """Budget-aware wrapper using jet.adapters.llama_cpp.factory."""

    def __init__(self):
        self._client = get_llm_client()
        self.tokens_used = 0
        self._grammars = {}

    def _load_grammar(self, name: str) -> str:
        if name not in self._grammars:
            path = os.path.join(GRAMMAR_DIR, f"{name}.gbnf")
            if not os.path.isfile(path):
                available = (
                    os.listdir(GRAMMAR_DIR) if os.path.isdir(GRAMMAR_DIR) else []
                )
                raise FileNotFoundError(
                    f"Grammar file not found: {path}\n"
                    f"GRAMMAR_DIR={GRAMMAR_DIR}\n"
                    f"Available files: {available}"
                )
            self._grammars[name] = open(path).read()
            logger.debug(f"Loaded grammar '{name}' from {path}")
        return self._grammars[name]

    async def chat(
        self,
        messages: list[dict],
        grammar: str | None = None,
        max_tokens: int = 1024,
    ) -> dict | str:
        kwargs: dict[str, Any] = {
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.95,
            "presence_penalty": 1.5,
            "seed": 42,
            "stream": True,
        }
        if grammar:
            grammar_content = self._load_grammar(grammar)
            logger.debug(
                f"Grammar '{grammar}' payload: length={len(grammar_content)}, "
                f"rule_count={grammar_content.count('::=')}, "
                f"enable_thinking=False"
            )
            kwargs["extra_body"] = {
                "grammar": grammar_content,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        else:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

        loop = asyncio.get_running_loop()
        stream = await loop.run_in_executor(
            None, lambda: self._client.chat.completions.create(**kwargs)
        )

        content_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        for chunk in stream:
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                content_parts.append(delta.content)
                print(delta.content, end="", flush=True)
        print()

        self.tokens_used += prompt_tokens + completion_tokens
        content = "".join(content_parts)

        if grammar and not content.strip():
            logger.error(
                f"EMPTY response with grammar '{grammar}'. "
                f"Prompt tokens: {prompt_tokens or 'unknown'}. "
                f"Verify enable_thinking=False is reaching the server."
            )
            return {"error": "EMPTY_RESPONSE", "raw": ""}

        if grammar:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Grammar output parse failed: {content[:200]}")
                return {"error": "PARSE_FAIL", "raw": content}

        return content


class LocalRetriever:
    """Uses jet.adapters.llama_cpp embed/rerank utils + ChromaDB with chunking."""

    def __init__(self):
        self.chroma = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        try:
            self.chroma.delete_collection("swarm_findings")
            logger.info("Deleted existing 'swarm_findings' collection")
        except ValueError:
            pass

        self.collection = self.chroma.get_or_create_collection(
            "swarm_findings",
            metadata={"hnsw:space": "cosine"},
            embedding_function=JetEmbeddingFunction(),
        )
        logger.info(
            "ChromaDB collection initialized with JetEmbeddingFunction "
            "(custom llama.cpp embeddings)"
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Async wrapper around jet embed() with batch dedup and VRAM safety."""
        if not texts:
            return []

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: embed(
                texts,
                model=DEFAULT_EMBED_MODEL,
                return_format="list",
                show_progress=True,
            ),
        )
        return result

    async def rerank_docs(
        self, query: str, docs: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[dict]:
        """Async wrapper around jet rerank() with token-aware truncation.

        The bge-rerank-v2-m3 model has a 1024 token limit. Documents exceeding
        this cause 500 errors. We use truncate_texts() to safely trim documents
        to fit within the reranker's context window.
        """
        if not docs:
            logger.debug("No documents to rerank")
            return []

        # Reranker model has 1024 token context window
        # Reserve 32 tokens for query + special tokens
        MAX_RERANK_TOKENS = 992  # 1024 - 32 buffer

        doc_texts = []
        for d in docs:
            text = d.get("text", d.get("content", ""))

            # Count actual tokens in the text
            token_count = count_tokens(text, model=DEFAULT_LLM_MODEL)

            if token_count > MAX_RERANK_TOKENS:
                logger.debug(
                    f"Document too long for reranker ({token_count} > {MAX_RERANK_TOKENS} tokens). "
                    f"Truncating with sentence awareness."
                )
                # Use smart truncation to preserve sentence boundaries
                text = truncate_texts(
                    text,
                    model=LLM_MODEL,
                    max_tokens=MAX_RERANK_TOKENS,
                    strict_sentences=True,
                    show_progress=False,
                )
                new_token_count = count_tokens(text, model=DEFAULT_LLM_MODEL)
                logger.debug(
                    f"Truncated for reranker: {token_count} → {new_token_count} tokens"
                )

            doc_texts.append(text)

        # Additional safety: double-check total batch size
        total_tokens = sum(count_tokens(t, model=DEFAULT_LLM_MODEL) for t in doc_texts)
        logger.debug(
            f"Reranking {len(doc_texts)} documents "
            f"(avg {total_tokens // max(len(doc_texts), 1)} tokens/doc, "
            f"{total_tokens} total)"
        )

        loop = asyncio.get_running_loop()
        try:
            rerank_results = await loop.run_in_executor(
                None,
                lambda: rerank(query, doc_texts, top_n=min(top_k, len(docs))),
            )
            logger.debug(f"Rerank successful: {len(rerank_results)} results")
            return [docs[r["index"]] for r in rerank_results]
        except Exception as e:
            logger.error(f"Rerank failed despite token-aware truncation: {e}")
            raise

    async def store_finding(self, finding: dict, use_chunking: bool = None):
        """Store finding with smart chunking for better retrieval.

        Key improvements:
        1. Stores FULL content (no pre-truncation) - let chunking handle sizing
        2. Uses truncate_texts() for LLM evaluation portions (sentence-aware)
        3. Evaluates chunk-level relevance independently
        4. Handles edge cases: empty content, embedding failures, chunking failures

        Args:
            finding: Finding dict with content, subtask_id, etc.
            use_chunking: If True, split content into semantic chunks.
                        If False, fall back to original single-chunk behavior.
                        If None, uses ENABLE_CHUNKING config.
        """
        if use_chunking is None:
            use_chunking = ENABLE_CHUNKING

        content = finding["content"]
        if not content:
            logger.warning(
                f"Empty content for subtask '{finding.get('subtask_id', 'unknown')}', "
                f"skipping storage"
            )
            return

        subtask_id = finding["subtask_id"]
        original_question = finding.get("question", "")
        finding_url = finding.get("url", "")

        # Pre-calculate token count for logging
        try:
            total_tokens = count_tokens(content, model=DEFAULT_LLM_MODEL)
        except Exception as e:
            logger.warning(
                f"Token counting failed for '{subtask_id}', "
                f"estimating from character length: {e}"
            )
            total_tokens = len(content) // 4  # Rough estimate

        metadata_base = {
            "url": finding_url,
            "branch": finding.get("branch_id", ""),
            "confidence": finding.get("confidence", "NONE"),
            "subtask_id": subtask_id,
            "original_question": original_question[:200],  # Truncate for metadata
        }

        if use_chunking:
            # Use sentence-aware chunking on FULL content
            logger.info(
                f"Chunking content for '{subtask_id}': "
                f"{len(content)} chars, ~{total_tokens} tokens"
            )

            try:
                chunks = chunk_texts_with_data(
                    content,
                    chunk_size=CHUNK_SIZE,  # 256
                    chunk_overlap=CHUNK_OVERLAP,  # 50 - but this needs buffer consideration
                    model=LLM_MODEL,
                    buffer=10,  # ADD: small buffer to help overlap work
                    strict_sentences=True,
                    min_chunk_size=MIN_CHUNK_SIZE,  # 64
                    show_progress=False,
                )
            except Exception as e:
                logger.error(
                    f"Chunking failed for '{subtask_id}', "
                    f"falling back to single chunk: {e}"
                )
                return await self.store_finding(finding, use_chunking=False)

            if not chunks:
                logger.warning(
                    f"No chunks generated for subtask '{subtask_id}' "
                    f"(content: {len(content)} chars, {total_tokens} tokens). "
                    f"Falling back to single chunk."
                )
                return await self.store_finding(finding, use_chunking=False)

            total_chunk_tokens = sum(c["num_tokens"] for c in chunks)
            avg_tokens = total_chunk_tokens // len(chunks)
            logger.info(
                f"Chunked finding '{subtask_id}' into {len(chunks)} pieces "
                f"(avg {avg_tokens} tokens/chunk, {total_chunk_tokens} total, "
                f"{chunks[0].get('overlap_start_idx') is not None and 'with' or 'no'} overlap)"
            )

            # Evaluate and store each chunk independently
            stored_count = 0
            skipped_count = 0

            for idx, chunk in enumerate(chunks):
                chunk_id = f"{subtask_id}_chunk_{idx}"

                # Quick relevance check for individual chunk
                chunk_confidence = self._evaluate_chunk_relevance(
                    subtask_id=subtask_id,
                    chunk_content=chunk["content"],
                    question=original_question,
                )

                # Skip chunks with very low relevance if we have enough
                if chunk_confidence == "LOW" and stored_count > 0:
                    logger.debug(
                        f"Skipping low-relevance chunk {idx + 1}/{len(chunks)} "
                        f"of '{subtask_id}' ({chunk['num_tokens']} tokens)"
                    )
                    skipped_count += 1
                    continue

                try:
                    # Embed the chunk
                    emb_list = await self.embed_texts([chunk["content"]])
                    if not emb_list or len(emb_list) == 0:
                        logger.error(
                            f"Embedding returned empty for chunk {idx + 1}/{len(chunks)} "
                            f"of '{subtask_id}' ({chunk['num_tokens']} tokens)"
                        )
                        continue

                    # Store with rich metadata including chunk-specific confidence
                    self.collection.upsert(
                        ids=[chunk_id],
                        embeddings=[emb_list[0]],
                        documents=[chunk["content"]],
                        metadatas=[
                            {
                                **metadata_base,
                                "chunk_index": idx,
                                "total_chunks": len(chunks),
                                "token_count": chunk["num_tokens"],
                                "chunk_start_idx": chunk["start_idx"],
                                "chunk_end_idx": chunk["end_idx"],
                                "has_overlap": bool(
                                    chunk.get("overlap_start_idx") is not None
                                    or chunk.get("overlap_end_idx") is not None
                                ),
                                "chunk_confidence": chunk_confidence,
                                "content_length": len(chunk["content"]),
                            }
                        ],
                    )
                    stored_count += 1
                    logger.debug(
                        f"Stored chunk {idx + 1}/{len(chunks)} of '{subtask_id}' "
                        f"({chunk['num_tokens']} tokens, "
                        f"confidence={chunk_confidence}, "
                        f"overlap={chunk.get('overlap_start_idx') is not None})"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to store chunk {idx + 1}/{len(chunks)} "
                        f"of '{subtask_id}': {type(e).__name__}: {e}"
                    )

            if stored_count == 0:
                logger.error(
                    f"No chunks successfully stored for '{subtask_id}'. "
                    f"Attempted {len(chunks)} chunks, skipped {skipped_count}."
                )
                # Fall back to single chunk to ensure we don't lose data
                logger.info(f"Attempting single-chunk fallback for '{subtask_id}'")
                return await self.store_finding(finding, use_chunking=False)
            else:
                logger.info(
                    f"Successfully stored {stored_count}/{len(chunks)} chunks "
                    f"for '{subtask_id}' "
                    f"(skipped {skipped_count} low-relevance chunks)"
                )

        else:
            # Original behavior with token-aware truncation
            max_embed_tokens = 500  # Typical embedding model limit

            if total_tokens > max_embed_tokens:
                logger.warning(
                    f"Content too long for single embedding "
                    f"({total_tokens} tokens > {max_embed_tokens} limit). "
                    f"Using sentence-aware truncation."
                )

                try:
                    content = truncate_texts(
                        content,
                        model=LLM_MODEL,
                        max_tokens=max_embed_tokens,
                        strict_sentences=True,
                        show_progress=False,
                    )
                    truncated_tokens = count_tokens(content, model=DEFAULT_LLM_MODEL)
                    logger.info(
                        f"Truncated content: {total_tokens} → {truncated_tokens} tokens "
                        f"({len(content)} chars)"
                    )
                except Exception as e:
                    logger.error(
                        f"Smart truncation failed for '{subtask_id}', "
                        f"using character-based fallback: {e}"
                    )
                    # Character-based fallback as last resort
                    char_limit = max_embed_tokens * 4  # Rough estimate
                    content = content[:char_limit] + "..."
                    truncated_tokens = len(content) // 4
            else:
                truncated_tokens = total_tokens

            if not content:
                logger.error(
                    f"Content became empty after truncation for '{subtask_id}'"
                )
                return

            try:
                emb_list = await self.embed_texts([content])
                if not emb_list or len(emb_list) == 0:
                    logger.error(
                        f"Embedding failed for subtask '{subtask_id}'. "
                        f"Content length: {len(content)} chars, "
                        f"{truncated_tokens} tokens. Skipping storage."
                    )
                    return

                self.collection.upsert(
                    ids=[subtask_id],
                    embeddings=[emb_list[0]],
                    documents=[content],
                    metadatas=[
                        {
                            **metadata_base,
                            "token_count": truncated_tokens,
                            "content_length": len(content),
                            "chunk_index": 0,
                            "total_chunks": 1,
                        }
                    ],
                )
                logger.info(
                    f"Stored single finding '{subtask_id}' "
                    f"({len(content)} chars, {truncated_tokens} tokens)"
                )

            except Exception as e:
                logger.error(
                    f"Failed to store single finding '{subtask_id}': "
                    f"{type(e).__name__}: {e}"
                )

    def _evaluate_chunk_relevance(
        self,
        subtask_id: str,
        chunk_content: str,
        question: str,
    ) -> str:
        """Estimate chunk relevance to the original question.

        Uses keyword overlap as a fast heuristic. For production,
        consider using a small classifier model or LLM-based evaluation.

        Args:
            subtask_id: The subtask this chunk belongs to
            chunk_content: The chunk text content
            question: The original question that prompted this search

        Returns:
            "HIGH", "MEDIUM", or "LOW" confidence level
        """
        if not question:
            logger.debug(
                f"No question provided for '{subtask_id}', "
                f"defaulting to MEDIUM confidence"
            )
            return "MEDIUM"

        # Normalize and tokenize
        question_words = set(
            word.lower().strip(".,!?;:()[]{}\"'-")
            for word in question.split()
            if len(word) > 2  # Skip very short words
        )
        chunk_words = set(
            word.lower().strip(".,!?;:()[]{}\"'-") for word in chunk_content.split()
        )

        if not question_words:
            return "MEDIUM"

        # Calculate Jaccard similarity
        intersection = question_words & chunk_words
        union = question_words | chunk_words
        similarity = len(intersection) / len(union) if union else 0

        # Also check for question terms appearing in chunk
        term_presence = sum(
            1 for term in question_words if term in chunk_content.lower()
        )
        coverage = term_presence / len(question_words)

        # Combined score
        combined_score = (similarity * 0.4) + (coverage * 0.6)

        if combined_score > 0.3:
            confidence = "HIGH"
        elif combined_score > 0.1:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        logger.debug(
            f"Chunk relevance for '{subtask_id}': "
            f"similarity={similarity:.3f}, coverage={coverage:.3f}, "
            f"combined={combined_score:.3f} → {confidence}"
        )

        return confidence

    async def recall(
        self, query: str, top_k: int = 3, branch_filter: str = None
    ) -> list[dict]:
        """Basic recall without chunk merging (for backward compatibility)."""
        where = {"branch": branch_filter} if branch_filter else None
        logger.debug(f"Recall query using registered custom embedding function")
        results = self.collection.query(
            query_texts=[query], n_results=top_k, where=where
        )

        if not results["documents"][0]:
            return []

        return [
            {"content": d, "url": m.get("url", ""), "score": s}
            for d, m, s in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    async def recall_with_chunks(
        self,
        query: str,
        top_k: int = 3,
        branch_filter: str = None,
        merge_chunks: bool = None,
        max_merged_tokens: int = 3000,  # ADD: limit merged content size
    ) -> list[dict]:
        """Enhanced recall that handles chunked storage.

        When findings are stored as multiple chunks, this method can:
        - Merge chunks from the same finding back together (merge_chunks=True)
        - Return individual chunks for granular retrieval (merge_chunks=False)

        Args:
            query: Search query text.
            top_k: Number of results to return.
            branch_filter: Optional branch ID filter.
            merge_chunks: If True, merge chunks from same finding.
                         If None, uses MERGE_CHUNKS_ON_RECALL config.
            max_merged_tokens: When merging chunks, limit total content tokens.

        Returns:
            List of dicts with content, url, score, subtask_id, and optionally
            chunk_index/total_chunks when merge_chunks=False.
        """
        if merge_chunks is None:
            merge_chunks = MERGE_CHUNKS_ON_RECALL

        where = {"branch": branch_filter} if branch_filter else None
        logger.debug(
            f"Recall query with chunking support "
            f"(merge_chunks={merge_chunks}, top_k={top_k})"
        )

        # Get more results since we might have multiple chunks per finding
        fetch_count = top_k * 3 if merge_chunks else top_k
        results = self.collection.query(
            query_texts=[query],
            n_results=fetch_count,
            where=where,
        )

        # Defensive: check for missing/empty results
        docs = results.get("documents", [None])
        metas = results.get("metadatas", [None])
        dists = results.get("distances", [None])
        if not docs or not docs[0]:
            logger.debug("No results found in recall_with_chunks")
            return []

        if not merge_chunks:
            # Return individual chunks
            chunk_results = [
                {
                    "content": d,
                    "url": m.get("url", ""),
                    "score": s,
                    "chunk_index": m.get("chunk_index"),
                    "total_chunks": m.get("total_chunks"),
                    "subtask_id": m.get("subtask_id"),
                    "token_count": m.get("token_count"),
                }
                for d, m, s in zip(docs[0], metas[0], dists[0])
            ]
            logger.debug(
                f"Returning {len(chunk_results)} individual chunks (no merging)"
            )
            return chunk_results[:top_k]

        # Merge chunks from the same finding
        findings_map: dict[str, dict] = {}
        for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
            subtask_id = str(meta.get("subtask_id", "unknown"))
            if subtask_id not in findings_map:
                findings_map[subtask_id] = {
                    "url": meta.get("url", ""),
                    "best_score": dist,
                    "chunks": [],
                    "subtask_id": subtask_id,
                    "total_chunks": meta.get("total_chunks", 1),
                    "confidence": meta.get("confidence", "NONE"),
                }

            # Keep best (lowest distance) score
            findings_map[subtask_id]["best_score"] = min(
                findings_map[subtask_id]["best_score"], dist
            )

            findings_map[subtask_id]["chunks"].append(
                {
                    "content": doc,
                    "score": dist,
                    "chunk_index": meta.get("chunk_index", 0),
                    "token_count": meta.get("token_count", 0),
                }
            )

        # Sort chunks by index and merge content progressively, respecting token limit
        merged_findings = []
        for finding in sorted(findings_map.values(), key=lambda x: x["best_score"])[
            :top_k
        ]:
            finding["chunks"].sort(key=lambda x: x["chunk_index"])

            # Build merged content, respecting max_merged_tokens
            merged_parts = []
            current_tokens = 0
            for chunk in finding["chunks"]:
                chunk_tokens = chunk.get("token_count", 0)
                # If adding this chunk would exceed max tokens AND we already have content, break
                if current_tokens + chunk_tokens > max_merged_tokens and merged_parts:
                    logger.debug(
                        f"Stopping chunk merge at {current_tokens}/{max_merged_tokens} tokens "
                        f"for '{finding['subtask_id']}' ({len(merged_parts)}/{len(finding['chunks'])} chunks)"
                    )
                    break
                merged_parts.append(chunk["content"])
                current_tokens += chunk_tokens

            # Merge the selected chunks
            merged_content = "\n\n".join(merged_parts)
            total_tokens = sum(c.get("token_count", 0) for c in finding["chunks"])

            merged_findings.append(
                {
                    "content": merged_content,
                    "url": finding["url"],
                    "score": finding["best_score"],
                    "subtask_id": finding["subtask_id"],
                    "num_chunks_merged": len(merged_parts),
                    "total_chunks": finding["total_chunks"],
                    "total_tokens": total_tokens,
                    "merged_tokens": current_tokens,  # ADD: actual tokens in merged content
                    "confidence": finding["confidence"],
                }
            )

        total_chunks_merged = sum(f["num_chunks_merged"] for f in merged_findings)
        logger.info(
            f"Recalled {len(merged_findings)} merged findings from "
            f"{total_chunks_merged} chunks "
            f"(scores: {[f'{f['score']:.3f}' for f in merged_findings]})"
        )

        return merged_findings


class BrowserManager:
    _ctx = None

    @classmethod
    async def get_context(cls):
        if cls._ctx is None:
            pw = await async_playwright().start()
            browser = await pw.chromium.launch(headless=True)
            cls._ctx = await browser.new_context(
                user_agent="Mozilla/5.0 (ResearchBot/1.0)",
                viewport={"width": 1280, "height": 800},
            )
        return cls._ctx


async def extract_page(url: str) -> dict:
    ctx = await BrowserManager.get_context()
    page = await ctx.new_page()
    try:
        await page.goto(url, timeout=15000, wait_until="domcontentloaded")
        html = await page.content()
        text = trafilatura.extract(html, include_comments=False) or ""
        # FIX: Don't truncate here, let chunking handle it
        return {
            "url": url,
            "text": text,  # Full text, no truncation
            "text_length": len(text),  # Metadata for logging
        }
    except Exception as e:
        logger.warning(f"Browser fail {url}: {e}")
        return {"url": url, "text": "", "error": str(e)}
    finally:
        await page.close()


class SwarmState(TypedDict):
    query: str
    subtasks: list[dict]
    findings: list[dict]
    iteration: int
    tokens_used: int
    start_time: float
    final_answer: str | None


class DedupCache:
    def __init__(self):
        import sqlite3

        self.conn = sqlite3.connect(CACHE_DB)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY, ts REAL)"
        )

    def is_seen(self, query: str, url: str = "") -> bool:
        h = hashlib.sha256(f"{query}|{url}".encode()).hexdigest()
        return (
            self.conn.execute("SELECT 1 FROM seen WHERE hash=?", (h,)).fetchone()
            is not None
        )

    def mark_seen(self, query: str, url: str = ""):
        h = hashlib.sha256(f"{query}|{url}".encode()).hexdigest()
        self.conn.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)", (h, time.time()))
        self.conn.commit()


async def web_search(query: str) -> list[str]:
    """Async wrapper around jet.search.searxng.search_searxng."""
    loop = asyncio.get_running_loop()
    try:
        results = await loop.run_in_executor(
            None,
            lambda: search_searxng(
                query=query,
                query_url=SEARXNG_QUERY_URL,
                count=SEARXNG_MAX_RESULTS,
                min_score=SEARXNG_MIN_SCORE,
                engines=SEARXNG_ENGINES,
                categories=SEARXNG_CATEGORIES,
                use_cache=SEARXNG_USE_CACHE,
            ),
        )
        urls = [r["url"] for r in results if r.get("url")]
        logger.info(f"SearXNG returned {len(urls)} URLs for: {query[:80]}")
        return urls
    except Exception as e:
        logger.error(f"SearXNG search failed for '{query[:80]}': {e}")
        return []


llm = LocalLLMClient()
retriever = LocalRetriever()
dedup = DedupCache()


async def _safe_llm_call(
    messages: list[dict], role: str, grammar: str = None
) -> dict | str:
    """Context degradation cascade using smart token-aware truncation."""
    budget = BUDGETS[role]

    # FIX: Use count_tokens with proper chat template detection
    # The issue is messages format doesn't match what apply_chat_template expects
    # Try adding add_generation_prompt parameter or use server-side counting
    try:
        # Attempt chat-specific counting first
        total = count_tokens(messages, model=DEFAULT_LLM_MODEL, use_server=True)
    except Exception:
        # Fall back to concatenated text counting (current behavior)
        combined = " ".join(msg.get("content", "") for msg in messages)
        total = count_tokens(combined, model=DEFAULT_LLM_MODEL)

    limit = sum(budget.values())

    if total <= limit:
        logger.debug(f"[{role}] Context OK ({total}/{limit} tokens)")
        return await llm.chat(messages, grammar=grammar, max_tokens=budget["output"])

    logger.warning(
        f"[{role}] Context overflow ({total}/{limit} tokens). "
        f"Trimming longest user message."
    )

    # Find the longest user message (usually contains the documents)
    user_msgs = [(i, m) for i, m in enumerate(messages) if m["role"] == "user"]
    user_msgs.sort(key=lambda x: len(x[1].get("content", "")), reverse=True)

    if user_msgs:
        idx, msg = user_msgs[0]
        content = msg["content"]

        # Calculate how many tokens we need to remove
        excess = total - limit
        current_tokens = count_tokens(content, model=DEFAULT_LLM_MODEL)
        target_tokens = max(current_tokens - excess, 100)  # Keep at least 100 tokens

        logger.debug(
            f"[{role}] Truncating user message from {current_tokens} to "
            f"~{target_tokens} tokens (need to remove {excess} tokens)"
        )

        # Use smart truncation that preserves sentence boundaries
        if target_tokens > 0 and current_tokens > target_tokens:
            trimmed = truncate_texts(
                content,
                model=LLM_MODEL,
                max_tokens=target_tokens,
                strict_sentences=True,
                show_progress=False,
            )
            messages[idx] = {
                "role": "user",
                "content": (
                    trimmed + "\n\n[Note: Content truncated to fit context window]"
                ),
            }

            new_tokens = count_tokens(trimmed, model=DEFAULT_LLM_MODEL)
            logger.info(
                f"[{role}] Truncated content: {len(content)} → "
                f"{len(trimmed)} chars "
                f"({current_tokens} → {new_tokens} tokens)"
            )

    return await llm.chat(messages, grammar=grammar, max_tokens=budget["output"])


async def planner_node(state: SwarmState) -> dict:
    existing = state.get("findings", [])
    history_summary = ""
    if existing:
        lines = [
            f"- [{f['subtask_id']}] {f.get('summary', f['content'][:80])}"
            for f in existing
        ]
        history_summary = "\n".join(lines)[: BUDGETS["planner"]["history"] * 4]

    messages = [
        {
            "role": "system",
            "content": (
                "You decompose research queries into subtasks. "
                "If prior findings exist, identify GAPS only. Output JSON per grammar."
            ),
        },
        {
            "role": "user",
            "content": (f"Query: {state['query']}\nPrior findings:\n{history_summary}"),
        },
    ]

    result = await _safe_llm_call(messages, "planner", grammar="planner")
    if isinstance(result, dict) and "error" in result:
        logger.error(f"Planner failed: {result}")
        return {"subtasks": state.get("subtasks", [])}

    new_tasks = result.get("subtasks", [])
    for t in new_tasks:
        t.setdefault("branch_id", f"branch_{state['iteration']}")

    return {
        "subtasks": state.get("subtasks", []) + new_tasks,
        "iteration": state.get("iteration", 0) + 1,
    }


async def searcher_node(state: SwarmState) -> dict:
    answered_ids = {f["subtask_id"] for f in state.get("findings", [])}
    task = next((t for t in state["subtasks"] if t["id"] not in answered_ids), None)
    if not task:
        logger.debug("No unanswered subtasks, skipping search")
        return {}

    # Use enhanced recall with chunk merging for deduplication check
    recalled = await retriever.recall_with_chunks(
        task["question"],
        top_k=1,
        merge_chunks=True,
    )

    if recalled and recalled[0].get("score", 1.0) < (1 - SEMANTIC_DEDUP_THRESHOLD):
        logger.info(
            f"Dedup hit for '{task['question'][:60]}' "
            f"(score: {recalled[0]['score']:.3f}, "
            f"merged {recalled[0].get('num_chunks_merged', 1)} chunks)"
        )
        return {
            "findings": state.get("findings", [])
            + [
                {
                    "subtask_id": task["id"],
                    "question": task["question"],
                    "content": recalled[0]["content"],
                    "url": recalled[0].get("url", ""),
                    "confidence": "RECALLED",
                    "summary": recalled[0]["content"][:100],
                    "branch_id": task.get("branch_id"),
                }
            ]
        }

    urls = await web_search(task["question"])
    candidates = [{"text": "", "url": u} for u in urls]
    ranked = await retriever.rerank_docs(task["question"], candidates)

    finding_content = ""
    finding_url = ""
    if ranked:
        page = await extract_page(ranked[0]["url"])
        finding_content = page["text"]
        finding_url = page["url"]

    dedup.mark_seen(task["question"], finding_url)

    # --- BEGIN CHANGE: Use token-aware truncation for evaluation ---
    evaluation_content = truncate_texts(
        finding_content,
        model=LLM_MODEL,
        max_tokens=1500,  # Budget for evaluation
        strict_sentences=True,
        show_progress=False,
    )

    conf_messages = [
        {
            "role": "system",
            "content": (
                "Evaluate if content answers the question. Output JSON per grammar."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {task['question']}\n"
                f"Content: {evaluation_content}"  # Token-aware truncation
            ),
        },
    ]
    # --- END CHANGE ---

    conf = await _safe_llm_call(conf_messages, "searcher", grammar="confidence")
    verdict = conf.get("verdict", "NONE") if isinstance(conf, dict) else "NONE"

    comp_messages = [
        {
            "role": "system",
            "content": ("Compress findings for child agents. Output JSON per grammar."),
        },
        {
            "role": "user",
            "content": (
                f"Compress for: '{task['question']}'\n{finding_content[:3000]}"
            ),
        },
    ]
    compressed = await _safe_llm_call(comp_messages, "compressor", grammar="compressor")
    summary = (
        compressed.get("summary", finding_content[:100])
        if isinstance(compressed, dict)
        else finding_content[:100]
    )

    new_finding = {
        "subtask_id": task["id"],
        "question": task["question"],
        "content": finding_content,
        "url": finding_url,
        "confidence": verdict,
        "summary": summary,
        "branch_id": task.get("branch_id"),
    }
    await retriever.store_finding(new_finding)

    return {"findings": state.get("findings", []) + [new_finding]}


async def synthesizer_node(state: SwarmState) -> dict:
    global_index = "\n".join(
        f"- [{f['subtask_id']}] ({f['confidence']}) {f.get('summary', '')}"
        for f in state.get("findings", [])
    )[: BUDGETS["synthesizer"]["global_index"] * 4]

    # Use recall_with_chunks for semantic retrieval
    recalled_findings = await retriever.recall_with_chunks(
        state["query"],
        top_k=5,
        merge_chunks=True,
    )

    if recalled_findings:
        # Rerank the recalled findings for better ordering
        top_findings = await retriever.rerank_docs(
            state["query"], recalled_findings, top_k=5
        )
        logger.info(
            f"Synthesizer using {len(top_findings)} recalled & reranked findings"
        )
    else:
        # Fallback: use findings directly from state
        logger.info("No recall hits, falling back to state findings for synthesis")
        top_findings = await retriever.rerank_docs(
            state["query"], state.get("findings", []), top_k=5
        )
        logger.info(f"Synthesizer using {len(top_findings)} findings from state")

    detailed = "\n---\n".join(f["content"][:2000] for f in top_findings)

    messages = [
        {
            "role": "system",
            "content": (
                "Synthesize a comprehensive, cited answer from findings. "
                "Acknowledge gaps honestly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original query: {state['query']}\n"
                f"Global index:\n{global_index}\n"
                f"Detailed findings:\n{detailed}"
            ),
        },
    ]
    answer = await _safe_llm_call(messages, "synthesizer")
    logger.info(
        f"Synthesizer returned: {type(answer)}, length: {len(str(answer)) if answer else 0}"
    )
    return {"final_answer": answer if isinstance(answer, str) else str(answer)}


def should_recurse(state: SwarmState) -> str:
    elapsed = time.time() - state.get("start_time", time.time())
    if (
        elapsed > MAX_WALL_SECONDS
        or llm.tokens_used > MAX_TOTAL_TOKENS
        or state.get("iteration", 0) >= MAX_ITERATIONS
    ):
        logger.warning(
            f"Budget exhausted. "
            f"Elapsed={elapsed:.0f}s "
            f"Tokens={llm.tokens_used} "
            f"Iter={state.get('iteration')}"
        )
        return "synthesize"

    answered = {f["subtask_id"] for f in state.get("findings", [])}
    unanswered = [t for t in state.get("subtasks", []) if t["id"] not in answered]
    partial = [f for f in state.get("findings", []) if f.get("confidence") == "PARTIAL"]

    if unanswered:
        max_d = max((t.get("depth", 0) for t in unanswered), default=0)
        if max_d < MAX_DEPTH:
            return "search"

    if partial:
        return "plan"

    if not unanswered:
        return "synthesize"

    return "synthesize"


async def run_swarm(query: str) -> str:
    graph = StateGraph(SwarmState)
    graph.add_node("plan", planner_node)
    graph.add_node("search", searcher_node)
    graph.add_node("synthesize", synthesizer_node)

    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", lambda _: "search", {"search": "search"})
    graph.add_conditional_edges(
        "search",
        should_recurse,
        {"search": "search", "plan": "plan", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)

    app = graph.compile(checkpointer=MemorySaver())
    initial_state = {
        "query": query,
        "subtasks": [],
        "findings": [],
        "iteration": 0,
        "tokens_used": 0,
        "start_time": time.time(),
        "final_answer": None,
    }

    result = await app.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": query[:50]}},
    )
    final_answer = result.get("final_answer", "No answer generated.")

    # ADD: Log what we got
    logger.info(
        f"Final answer length: {len(final_answer) if final_answer else 0} chars"
    )
    if not final_answer or final_answer == "No answer generated.":
        logger.error(f"Empty final answer. State keys: {list(result.keys())}")
        logger.error(f"Findings count: {len(result.get('findings', []))}")

    return final_answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run WebSwarm with a query.")
    parser.add_argument(
        "query",
        nargs="?",
        default=(
            "What are the supply chain risks for solid-state batteries in SE Asia?"
        ),
        help="The query to run WebSwarm on.",
    )
    args = parser.parse_args()

    answer = asyncio.run(run_swarm(args.query))
    print("\n" + "=" * 80 + "\n" + answer)
