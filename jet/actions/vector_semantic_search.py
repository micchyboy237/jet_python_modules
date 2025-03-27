from jet.llm.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_EMBEDDING_TOKENS
from jet.logger import logger, time_it

from typing import List, Dict, Union


class VectorSemanticSearch:
    def __init__(
        self,
        candidates: list[str],
        *,
        embed_model: OLLAMA_EMBED_MODELS = "mxbai-embed-large",
        reranking_model: OLLAMA_EMBED_MODELS = "all-minilm:33m",
    ):
        self.candidates = candidates
        self.model = None
        self.cross_encoder = None
        self.tokenized_paths = [path.split('.') for path in candidates]
        self.graph = None
        self.embed_model = embed_model
        self.reranking_model = reranking_model

    def get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L12-v2')
        return self.model

    def get_cross_encoder(self):
        if self.cross_encoder is None:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(
                'cross-encoder/ms-marco-TinyBERT-L-6')
        return self.cross_encoder

    def get_reranking_model(self):
        if self.reranking_model is None:
            from sentence_transformers import CrossEncoder
            self.reranking_model = CrossEncoder(
                'cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        return self.reranking_model

    def get_graph(self):
        if self.graph is None:
            import networkx as nx
            self.graph = nx.Graph()
        return self.graph

    def search(self, query: str | list[str]) -> Dict[str, List[Dict[str, float]]]:
        queries = query
        if isinstance(query, str):
            queries = [query]

        search_results = self.faiss_search(queries)
        return search_results

    @time_it
    def vector_based_search(self, queries: list[str]) -> Dict[str, List[Dict[str, float]]]:
        """Performs a batch search using vector similarity."""
        from sentence_transformers import util

        model = self.get_model()

        data_embeddings = model.encode(
            self.candidates, convert_to_tensor=True, clean_up_tokenization_spaces=True)
        query_embeddings = model.encode(
            queries, convert_to_tensor=True, clean_up_tokenization_spaces=True)

        results = {}
        for query, q_emb in zip(queries, query_embeddings):
            scores = util.cos_sim(q_emb, data_embeddings)[0].cpu().numpy()
            sorted_results = sorted(
                zip(self.candidates, scores), key=lambda x: x[1], reverse=True
            )
            results[query] = [{'text': text, 'score': score}
                              for text, score in sorted_results]

        return results

    @time_it
    def faiss_search(self, queries: list[str]) -> Dict[str, List[Dict[str, float]]]:
        """Performs FAISS-based batch search for efficiency."""
        import faiss
        from jet.actions import faiss_search

        top_k = len(self.candidates)
        nlist = None

        results = faiss_search(queries, self.candidates,
                               top_k=top_k, nlist=nlist)

        sorted_results = {query_line: sorted(res, key=lambda x: x['score'], reverse=True)
                          for query_line, res in results.items()}

        return sorted_results

    @time_it
    def annoy_search(self, queries: str | list[str]) -> Dict[str, List[Dict[str, float]]]:
        """Executes a annoy-based retrieval, aggregating search results."""
        from llama_index.core.schema import Document
        from jet.llm.ollama.constants import OLLAMA_SMALL_EMBED_MODEL, OLLAMA_LARGE_EMBED_MODEL
        from jet.llm.query import setup_index

        if isinstance(queries, str):
            queries = [queries]

        documents = [Document(text=candidate) for candidate in self.candidates]

        mode = "annoy"
        chunk_size = 256
        chunk_overlap = 40
        score_threshold = 0.2
        top_k = None
        embed_model = OLLAMA_SMALL_EMBED_MODEL

        query_nodes = setup_index(
            documents,
            mode=mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_model=embed_model,
        )

        results_dict = {}
        for query in queries:
            result = query_nodes(
                query, score_threshold=score_threshold, top_k=top_k)

            results = [{"text": node.text, "score": node.score}
                       for node in result["nodes"]]
            results_dict[query] = sorted(
                results, key=lambda x: x['score'], reverse=True)

        return results_dict

    @time_it
    def bm25_search(self, queries: str | list[str]) -> Dict[str, List[Dict[str, float]]]:
        """Executes a bm25-based retrieval, aggregating search results."""
        from llama_index.core.schema import Document
        from jet.llm.ollama.constants import OLLAMA_SMALL_EMBED_MODEL, OLLAMA_LARGE_EMBED_MODEL
        from jet.llm.query import setup_index

        if isinstance(queries, str):
            queries = [queries]

        documents = [Document(text=candidate) for candidate in self.candidates]

        mode = "bm25"
        chunk_size = 256
        chunk_overlap = 40
        score_threshold = 0.2
        top_k = None
        embed_model = OLLAMA_SMALL_EMBED_MODEL

        query_nodes = setup_index(
            documents,
            mode=mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_model=embed_model,
        )

        results_dict = {}
        for query in queries:
            result = query_nodes(
                query, score_threshold=score_threshold, top_k=top_k)

            results = [{"text": node.text, "score": node.score}
                       for node in result["nodes"]]
            results_dict[query] = sorted(
                results, key=lambda x: x['score'], reverse=True)

        return results_dict

    @time_it
    def fusion_search(self, queries: str | list[str]) -> Dict[str, List[Dict[str, float]]]:
        """Executes a fusion-based retrieval, aggregating search results."""
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
        from llama_index.core.schema import Document
        from jet.llm.ollama.constants import OLLAMA_SMALL_EMBED_MODEL, OLLAMA_LARGE_EMBED_MODEL
        from jet.llm.query import setup_index

        if isinstance(queries, str):
            queries = [queries]

        documents = [Document(text=candidate) for candidate in self.candidates]

        mode = "fusion"
        chunk_size = OLLAMA_MODEL_EMBEDDING_TOKENS[self.embed_model]
        chunk_overlap = 40
        score_threshold = 0.2
        top_k = None
        embed_model = self.embed_model

        query_nodes = setup_index(
            documents,
            mode=mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_model=embed_model,
        )

        results_dict = {}
        for query in queries:
            result = query_nodes(
                query, fusion_mode=FUSION_MODES.RELATIVE_SCORE, score_threshold=score_threshold, top_k=top_k)

            results = [{"text": node.text, "score": node.score}
                       for node in result["nodes"]]
            results_dict[query] = sorted(
                results, key=lambda x: x['score'], reverse=True)

        return results_dict

    @time_it
    def rerank_search(self, queries: list[str]) -> Dict[str, List[Dict[str, float]]]:
        from jet.llm.helpers.semantic_search import RerankerRetriever

        top_k = 10
        rerank_threshold = 0.3
        use_ollama = False

        # Initialize the retriever
        retriever = RerankerRetriever(
            data=self.candidates,
            use_ollama=use_ollama,
            collection_name="example_collection",
            embed_batch_size=32,
            overwrite=True
        )

        def search_with_reranking(query: Union[str, List[str]], top_k: int, rerank_threshold: float):
            if isinstance(query, str):  # If the query is a single string
                reranked_results = retriever.search_with_reranking(
                    query, top_k=top_k, rerank_threshold=rerank_threshold)
            elif isinstance(query, list):  # If the query is a list of strings
                reranked_results = [
                    retriever.search_with_reranking(q, top_k=top_k, rerank_threshold=rerank_threshold) for q in query
                ]

            return reranked_results

        # Perform search with reranking
        search_results_with_reranking = search_with_reranking(
            queries, top_k=top_k, rerank_threshold=rerank_threshold)

        # Organize results in the same format as other search methods
        sorted_results = {
            query_line: [{'text': res['document'], 'score': res['score']}
                         for res in sorted(res_list, key=lambda x: x['score'], reverse=True)]
            for query_line, res_list in zip(queries, search_results_with_reranking)
        }

        return sorted_results

    @time_it
    def graph_based_search(self, queries: list[str]) -> Dict[str, List[Dict[str, float]]]:
        import networkx as nx

        graph = self.get_graph()
        graph.add_nodes_from(self.candidates)

        model = self.get_model()
        query_embeddings = [model.encode(
            q, convert_to_tensor=True, clean_up_tokenization_spaces=True).cpu().numpy() for q in queries]
        embeddings = model.encode(self.candidates, convert_to_tensor=True,
                                  clean_up_tokenization_spaces=True).cpu().numpy()

        from sentence_transformers import util
        similarities = [util.cos_sim(q_emb, embeddings)[
            0].cpu().numpy() for q_emb in query_embeddings]

        for query_emb, similarity in zip(query_embeddings, similarities):
            query_emb_tuple = tuple(query_emb)
            for i, path in enumerate(self.candidates):
                graph.add_edge(query_emb_tuple, path, weight=similarity[i])

        pagerank_scores = nx.pagerank(graph, weight='weight')

        # Sort pagerank results by score
        sorted_pagerank_results = {query_line: [{'text': path, 'score': pagerank_scores[path]}
                                                for path in sorted(self.candidates, key=lambda p: pagerank_scores[p], reverse=True)]
                                   for query_line in queries}

        return sorted_pagerank_results

    @time_it
    def cross_encoder_search(self, queries: list[str]) -> Dict[str, List[Dict[str, float]]]:

        cross_encoder = self.get_cross_encoder()

        # Generate pairs of query and candidate paths
        pairs = [(q, path) for q in queries for path in self.candidates]

        # Predict the similarity scores
        scores = cross_encoder.predict(pairs)

        # Check if the number of scores matches the expected length
        if len(scores) != len(pairs):
            raise ValueError(f"Mismatch between number of score predictions ({
                             len(scores)}) and pairs ({len(pairs)})")

        # Organize the results by query line
        results = {}
        idx = 0
        for query_line in queries:
            # Get the scores for this query
            query_scores = scores[idx:idx + len(self.candidates)]
            results[query_line] = [
                {'text': self.candidates[i], 'score': query_scores[i]} for i in range(len(self.candidates))]
            idx += len(self.candidates)

        # Sort the results for each query line by score
        sorted_results = {query_line: sorted(res, key=lambda x: x['score'], reverse=True)
                          for query_line, res in results.items()}

        return sorted_results

    @staticmethod
    def parallel_tokenize(paths):
        """Tokenizes paths using multiprocessing."""
        from multiprocessing import Pool
        with Pool(processes=4) as pool:
            chunk_size = len(paths) // 4
            chunks = [paths[i:i + chunk_size]
                      for i in range(0, len(paths), chunk_size)]
            tokenized_chunks = pool.map(
                lambda x: [p.split('.') for p in x], chunks)
            return [item for sublist in tokenized_chunks for item in sublist]
