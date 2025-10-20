from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import stanza
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from markdown import markdown

class RAGContextImprover:
    def __init__(self, lang: str = 'en', embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the improver with Stanza pipeline and BERTopic using SentenceTransformer.
        
        Args:
            lang: Language for Stanza pipeline (default: 'en').
            embedding_model: SentenceTransformer model name (default: 'all-MiniLM-L6-v2').
        """
        self.nlp = stanza.Pipeline(lang=lang, processors='tokenize,ner')
        self.embedding_model = SentenceTransformer(embedding_model, device='cpu')  # CPU for Windows; M1 auto-optimizes
        self.topic_model = BERTopic(embedding_model=self.embedding_model, calculate_probabilities=True)

    def preprocess_documents(self, documents: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Converts markdown docs to text, chunks into sentences using Stanza, and extracts NER entities.
        Returns chunks and a doc_index to entities mapping.
        """
        chunks = []
        entity_map = {}  # chunk_index: [entities]
        for doc_idx, md_doc in enumerate(documents):
            text = self._markdown_to_text(md_doc)
            doc = self.nlp(text)
            for sent_idx, sentence in enumerate(doc.sentences):
                chunk = sentence.text
                chunks.append(chunk)
                chunk_idx = len(chunks) - 1
                entities = [ent.text for ent in sentence.ents]
                if entities:
                    entity_map[chunk_idx] = entities
        return chunks, entity_map

    def _markdown_to_text(self, md: str) -> str:
        """Converts markdown to plain text, preserving structure minimally."""
        html = markdown(md)
        return ''.join([line.strip() for line in html.split('\n') if line.strip()])

    def model_topics(self, chunks: List[str]) -> Dict[int, int]:
        """
        Fits BERTopic on chunks and returns chunk_index to topic mapping.
        Uses zero-shot if topics predefined; here, default clustering.
        """
        topics, _ = self.topic_model.fit_transform(chunks)
        return {i: topic for i, topic in enumerate(topics)}

    def retrieve_contexts(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        """
        Main method: Preprocesses docs, models topics, extracts query entities,
        filters chunks by topic similarity and entity match, returns top_k contexts.
        """
        chunks, entity_map = self.preprocess_documents(documents)
        topic_map = self.model_topics(chunks)

        # Process query with Stanza for entities
        query_doc = self.nlp(query)
        query_entities = {ent.text for ent in query_doc.ents}

        # Embed query using SentenceTransformer
        query_embedding = self.embedding_model.encode([query], batch_size=1, show_progress_bar=False)[0]

        # Score chunks: topic relevance + entity overlap
        scores = []
        for idx, chunk in enumerate(chunks):
            chunk_embedding = self.embedding_model.encode([chunk], batch_size=1, show_progress_bar=False)[0]
            sim = np.dot(query_embedding, chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
            entity_score = len(query_entities.intersection(set(entity_map.get(idx, [])))) / max(len(query_entities), 1)
            total_score = sim + entity_score  # Weighted sum; adjustable
            scores.append((idx, total_score))

        # Filter by relevant topics
        query_topic, _ = self.topic_model.transform([query])
        relevant_chunks = [idx for idx in range(len(chunks)) if topic_map[idx] == query_topic[0] or scores[idx][1] > 0.5]

        # Sort and select top_k
        top_indices = sorted(relevant_chunks, key=lambda idx: scores[idx][1], reverse=True)[:top_k]
        return [chunks[i] for i in top_indices]

    def search_documents(self, query: str, documents: List[str], top_k: int = 5, alpha: float = 0.5) -> List[str]:
        """
        Performs hybrid search (keyword + semantic) on documents, returning top_k relevant chunks.
        
        Args:
            query: Search query string.
            documents: List of documents (markdown or plain text).
            top_k: Number of top results to return.
            alpha: Fusion weight (0.0: keyword-only, 1.0: semantic-only, default 0.5).
        
        Returns:
            List of top_k ranked chunks.
        """
        chunks, _ = self.preprocess_documents(documents)  # Chunks via Stanza; ignore entities for pure search
        
        if not chunks:
            return []
        
        # Keyword search: TF-IDF cosine similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([query] + chunks)
        query_tfidf = tfidf_matrix[0:1]
        doc_tfidf = tfidf_matrix[1:]
        keyword_scores = (query_tfidf @ doc_tfidf.T).toarray().flatten()  # Cosine sim
        
        # Semantic search: Embeddings
        query_embedding = self.embedding_model.encode([query], batch_size=1, show_progress_bar=False)[0]
        chunk_embeddings = self.embedding_model.encode(chunks, batch_size=32, show_progress_bar=False)  # Batch for efficiency
        semantic_scores = np.array([
            np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            for emb in chunk_embeddings
        ])
        
        # Normalize scores to [0,1] for fusion
        keyword_norm = (keyword_scores - keyword_scores.min()) / (keyword_scores.max() - keyword_scores.min() + 1e-8)
        semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
        
        # Weighted fusion
        fused_scores = alpha * semantic_norm + (1 - alpha) * keyword_norm
        
        # Rank by score (descending)
        ranked_indices = np.argsort(fused_scores)[::-1][:top_k]
        return [chunks[i] for i in ranked_indices]