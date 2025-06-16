from typing import List, TypedDict, Optional
import urllib.parse
from jet.llm.mlx.mlx_types import EmbedModelType
from jet.llm.mlx.models import resolve_model
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import torch

device = "cpu"


class Link(TypedDict):
    url: str
    text: Optional[str]


class LinkSearchResult(TypedDict):
    url: str
    text: Optional[str]
    score: float


def search_links(links: List[Link], query: str, model: EmbedModelType = "all-MiniLM-L12-v2", top_k: int = 10, embedding_threshold: float = 0.3) -> List[LinkSearchResult]:
    """
    Search a list of links based on a query using a hybrid BM25 + embeddings approach.
    Links may include optional 'text' for additional context.
    Returns results with relevance scores.

    Args:
        links: List of dictionaries with 'url' (required) and 'text' (optional).
        query: Search query string.
        top_k: Number of BM25 candidates to consider for re-ranking (default: 10).
        embedding_threshold: Minimum cosine similarity for embeddings (default: 0.3).

    Returns:
        List of dictionaries with 'url', 'text' (if provided), and 'score'.
    """
    query = query.lower().strip() if query else ""
    if not query:
        return []

    noisy_params = {'session', 'token', 'id', 'sid', 'track'}

    def extract_url_content(url: str) -> str:
        """
        Extract searchable content from URL, decoding path, query params, and fragment.
        """
        try:
            parsed = urllib.parse.urlparse(url)
            path = parsed.path.strip('/')
            query_params = urllib.parse.parse_qs(parsed.query)
            query_str = ' '.join(
                f"{k} {v}" for k, values in query_params.items()
                for v in values if k.lower() not in noisy_params
            )
            fragment = parsed.fragment
            content = f"{path} {query_str} {fragment}".lower().strip()
            return ' '.join(content.split())
        except ValueError:
            return url.lower().strip()

    def extract_content(link: Link) -> str:
        """
        Combine URL content and optional text context.
        """
        url_content = extract_url_content(link['url'])
        text_content = link.get('text', '').lower().strip()
        content = f"{url_content} {text_content}".strip()
        return ' '.join(content.split())

    # # Preprocess for BM25
    # corpus = [extract_content(link).split() for link in links]
    # bm25 = BM25Okapi(corpus)
    # tokenized_query = query.lower().split()
    # bm25_scores = bm25.get_scores(tokenized_query)
    # bm25_results = [
    #     (links[i], score) for i, score in enumerate(bm25_scores) if score > 0
    # ]
    # bm25_results = sorted(
    #     bm25_results, key=lambda x: x[1], reverse=True)[:top_k]
    # candidate_links = [link for link, _ in bm25_results]

    # if not candidate_links:
    #     return []

    # Re-rank with embeddings
    model = SentenceTransformer(resolve_model(model), device=device)
    candidate_contents = [extract_content(link) for link in links]
    candidate_embeddings = model.encode(
        candidate_contents, batch_size=32, convert_to_tensor=True, show_progress_bar=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]

    # Combine results with scores
    results = [
        LinkSearchResult(
            url=links[i]['url'],
            text=links[i].get('text'),
            score=float(score.item())
        )
        for i, score in enumerate(cos_scores)
        if score > embedding_threshold
    ]
    return sorted(results, key=lambda x: x['score'], reverse=True)
