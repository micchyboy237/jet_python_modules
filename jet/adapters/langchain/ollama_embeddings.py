from typing import List
from langchain_ollama import OllamaEmbeddings as BaseOllamaEmbeddings
from jet.logger import logger
from tqdm import tqdm


class OllamaEmbeddings(BaseOllamaEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs in batches with progress tracking."""
        if not self._client:
            msg = (
                "Ollama client is not initialized. "
                "Please ensure Ollama is running and the model is loaded."
            )
            raise ValueError(msg)

        batch_size = 50  # Adjust based on experimentation
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding {batch_size} documents", unit="batch"):
            batch = texts[i:i + batch_size]
            try:
                response = self._client.embed(
                    self.model, batch, options=self._default_params, keep_alive=self.keep_alive
                )
                embeddings.extend(response["embeddings"])
            except Exception as e:
                logger.error(
                    f"Error embedding batch {i//batch_size + 1}: {str(e)}")
                raise
        return embeddings
