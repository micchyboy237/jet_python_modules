import chromadb
from config import PERSIST_DIR
from sentence_transformers import SentenceTransformer


class LongTermMemory:
    def __init__(self, persist_dir: str = f"{PERSIST_DIR}/agent_longterm_db"):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="smolagent_facts", metadata={"hnsw:space": "cosine"}
        )

    def add_fact(
        self, content: str, step_number: int, run_id: str = "default_run"
    ) -> str:
        if not content.strip():
            return "Empty fact ignored."
        embedding = self.embedder.encode(content).tolist()
        fact_id = f"fact-{run_id}-{step_number}-{self.collection.count()}"
        self.collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[{"step": step_number, "run_id": run_id, "type": "fact"}],
            ids=[fact_id],
        )
        return f"Fact saved (id: {fact_id})"

    def search(self, query: str, n_results: int = 6) -> str:
        if not query.strip():
            return "No query provided."
        emb = self.embedder.encode(query).tolist()
        res = self.collection.query(query_embeddings=[emb], n_results=n_results)
        if not res["documents"] or not res["documents"][0]:
            return "No relevant long-term facts found."
        lines = []
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            lines.append(
                f"• {doc} (step {meta['step']}, run {meta.get('run_id', '?')})"
            )
        return "\n".join(lines)


# Singleton / global instance (common pattern for agents)
long_term_memory = LongTermMemory()
