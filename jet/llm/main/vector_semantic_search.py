from jet.logger import logger, time_it


from typing import List, Dict, Union


class VectorSemanticSearch:
    def __init__(self, candidates: list[str]):
        self.candidates = candidates
        self.model = None
        self.cross_encoder = None
        self.tokenized_paths = [path.split('.') for path in candidates]
        self.graph = None
        self.embeddings = None
        self.reranking_model = None

    def get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
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

    @time_it
    def vector_based_search(self, query: str) -> Dict[str, List[Dict[str, float]]]:
        query = query.splitlines()
        model = self.get_model()
        query_embeddings = [model.encode(
            q, convert_to_tensor=True, clean_up_tokenization_spaces=True) for q in query]
        self.embeddings = model.encode(
            self.candidates, convert_to_tensor=True, clean_up_tokenization_spaces=True)
        from sentence_transformers import util
        scores = [util.cos_sim(q_emb, self.embeddings)[0].cpu().numpy()
                  for q_emb in query_embeddings]
        results = [(self.candidates[i], score)
                   for score_list in scores for i, score in enumerate(score_list)]

        # Sort results in reverse order of score
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        # Return in faiss_search format
        return {query_line: [{'text': sorted_results[i][0], 'score': sorted_results[i][1]} for i in range(len(sorted_results))]
                for query_line in query}

    @time_it
    def faiss_search(self, query: str) -> Dict[str, List[Dict[str, float]]]:
        import faiss
        from jet.llm.main import faiss_search

        queries = query.splitlines()
        top_k = 3
        nlist = 100

        results = faiss_search(queries, self.candidates,
                               top_k=top_k, nlist=nlist)

        # Sort results in reverse order of score
        sorted_results = {query_line: sorted(res, key=lambda x: x['score'], reverse=True)
                          for query_line, res in results.items()}

        return sorted_results

    @time_it
    def rerank_search(self, query: str) -> Dict[str, List[Dict[str, float]]]:
        from jet.llm.helpers.semantic_search import RerankerRetriever

        data = query.splitlines()

        query = "Sample document"  # Consider removing this line if it's a placeholder
        top_k = 10
        rerank_threshold = 0.3
        use_ollama = False

        # Initialize the retriever
        retriever = RerankerRetriever(
            data=data,
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
            data, top_k=top_k, rerank_threshold=rerank_threshold)

        # Organize results in the same format as other search methods
        sorted_results = {query_line: sorted(res, key=lambda x: x['score'], reverse=True)
                          for query_line, res in zip(data, search_results_with_reranking)}

        return sorted_results

    @time_it
    def graph_based_search(self, query: str) -> Dict[str, List[Dict[str, float]]]:
        import networkx as nx
        query = query.splitlines()
        graph = self.get_graph()
        graph.add_nodes_from(self.candidates)

        model = self.get_model()
        query_embeddings = [model.encode(
            q, convert_to_tensor=True, clean_up_tokenization_spaces=True).cpu().numpy() for q in query]
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
                                   for query_line in query}

        return sorted_pagerank_results

    @time_it
    def cross_encoder_search(self, query: str) -> Dict[str, List[Dict[str, float]]]:
        query = query.splitlines()
        cross_encoder = self.get_cross_encoder()

        # Generate pairs of query and candidate paths
        pairs = [(q, path) for q in query for path in self.candidates]

        # Predict the similarity scores
        scores = cross_encoder.predict(pairs)

        # Check if the number of scores matches the expected length
        if len(scores) != len(pairs):
            raise ValueError(f"Mismatch between number of score predictions ({
                             len(scores)}) and pairs ({len(pairs)})")

        # Organize the results by query line
        results = {}
        idx = 0
        for query_line in query:
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
        from multiprocessing import Pool
        with Pool(processes=4) as pool:
            chunk_size = len(paths) // 4
            chunks = [paths[i:i + chunk_size]
                      for i in range(0, len(paths), chunk_size)]
            tokenized_chunks = pool.map(
                lambda x: [p.split('.') for p in x], chunks)
            return [item for sublist in tokenized_chunks for item in sublist]
