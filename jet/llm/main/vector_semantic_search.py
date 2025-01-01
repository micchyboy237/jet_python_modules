from jet.logger import logger


class VectorSemanticSearch:
    def __init__(self, module_paths):
        self.module_paths = module_paths
        self.model = None
        self.cross_encoder = None
        self.tokenized_paths = [path.split('.') for path in module_paths]
        self.graph = None
        self.embeddings = None
        self.bm25 = None

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

    def get_bm25(self):
        if self.bm25 is None:
            from jet.llm.helpers.semantic_search import VectorSearchRetriever

            retriever = VectorSearchRetriever(
                use_ollama=True,
                use_reranker=True
            )
            self.bm25 = retriever
        return self.bm25

    def get_graph(self):
        if self.graph is None:
            import networkx as nx
            self.graph = nx.Graph()
        return self.graph

    def vector_based_search(self, query):
        model = self.get_model()
        query_embedding = model.encode(query, convert_to_tensor=True)
        self.embeddings = model.encode(
            self.module_paths, convert_to_tensor=True)
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, self.embeddings)[
            0].cpu().numpy()
        return sorted(zip(self.module_paths, scores), key=lambda x: x[1], reverse=True)

    def faiss_search(self, query):
        model = self.get_model()
        if self.embeddings is None:
            self.embeddings = model.encode(
                self.module_paths, convert_to_tensor=True).cpu().numpy()
        import numpy as np
        import torch
        if isinstance(self.embeddings, torch.Tensor):
            # Move tensor to CPU if it's on a GPU, then convert to numpy array
            self.embeddings = self.embeddings.cpu().detach().numpy()
        else:
            # Ensure embeddings are NumPy arrays
            self.embeddings = np.array(self.embeddings)
        import faiss
        d = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(self.embeddings)
        query_embedding = model.encode(
            [query], convert_to_tensor=True).cpu().numpy()
        distances, indices = index.search(query_embedding, 3)
        return [(self.module_paths[i], distances[0][idx]) for idx, i in enumerate(indices[0])]

    def bm25_search(self, query):
        bm25 = self.get_bm25()
        search_results = bm25.search_with_reranking(query)
        return [(result['document'], result['score']) for result in search_results]

    def graph_based_search(self, query):
        graph = self.get_graph()
        graph.add_nodes_from(self.module_paths)

        # Example: Update edge weights based on query similarity
        model = self.get_model()
        query_embedding = model.encode(
            query, convert_to_tensor=True).cpu().numpy()
        embeddings = model.encode(
            self.module_paths, convert_to_tensor=True).cpu().numpy()

        from sentence_transformers import util
        # Compute similarities between query and module paths
        similarities = util.cos_sim(query_embedding, embeddings)[
            0].cpu().numpy()

        # Update edges based on similarity scores
        for i, path in enumerate(self.module_paths):
            graph.add_edge(query, path, weight=similarities[i])

        import networkx as nx
        pagerank_scores = nx.pagerank(graph, weight='weight')
        return sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

    def cross_encoder_search(self, query):
        cross_encoder = self.get_cross_encoder()
        pairs = [(query, path) for path in self.module_paths]
        scores = cross_encoder.predict(pairs)
        return sorted(zip(self.module_paths, scores), key=lambda x: x[1], reverse=True)

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
