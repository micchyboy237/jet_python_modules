from jet.logger import logger


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
    def vector_based_search(self, query):
        # Split query into lines, each treated as a separate item
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
        return sorted(results, key=lambda x: x[1], reverse=True)

    @time_it
    def faiss_search(self, query):
        import faiss
        from jet.llm.main import faiss_search

        # Split query into lines, each treated as a separate item
        queries = query.splitlines()

        top_k = 3
        nlist = 100

        results = faiss_search(queries, self.candidates,
                               top_k=top_k, nlist=nlist)
        return results

    @time_it
    def rerank_search(self, query):
        # Split query into lines, each treated as a separate item
        query = query.splitlines()
        cross_encoder = self.get_reranking_model()
        pairs = [(q, path) for q in query for path in self.candidates]
        scores = cross_encoder.predict(pairs)
        return sorted(zip(self.candidates, scores), key=lambda x: x[1], reverse=True)

    @time_it
    def graph_based_search(self, query):
        import numpy as np
        from sentence_transformers import util
        import networkx as nx

        # Split query into lines, each treated as a separate item
        query = query.splitlines()
        graph = self.get_graph()
        graph.add_nodes_from(self.candidates)

        model = self.get_model()
        query_embeddings = [model.encode(
            q, convert_to_tensor=True, clean_up_tokenization_spaces=True).cpu().numpy() for q in query]
        embeddings = model.encode(self.candidates, convert_to_tensor=True,
                                  clean_up_tokenization_spaces=True).cpu().numpy()

        # Calculate similarities
        similarities = [util.cos_sim(q_emb, embeddings)[
            0].cpu().numpy() for q_emb in query_embeddings]

        # Update edges based on similarity scores
        for query_emb, similarity in zip(query_embeddings, similarities):
            query_emb_tuple = tuple(query_emb)  # Convert numpy array to tuple
            for i, path in enumerate(self.candidates):
                graph.add_edge(query_emb_tuple, path, weight=similarity[i])

        # Calculate page rank
        pagerank_scores = nx.pagerank(graph, weight='weight')

        # Convert non-string keys to a readable string format
        results = []
        for key, value in pagerank_scores.items():
            if isinstance(key, str):
                results.append((key, value))
            else:
                # Convert numpy array to string
                results.append((np.array_repr(key), value))

        return sorted(results, key=lambda x: x[1], reverse=True)

    @time_it
    def cross_encoder_search(self, query):
        # Split query into lines, each treated as a separate item
        query = query.splitlines()
        cross_encoder = self.get_cross_encoder()
        pairs = [(q, path) for q in query for path in self.candidates]
        scores = cross_encoder.predict(pairs)
        return sorted(zip(self.candidates, scores), key=lambda x: x[1], reverse=True)

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
