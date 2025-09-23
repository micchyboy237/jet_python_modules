import logging
from typing import Dict, List, Optional
from pydantic import ConfigDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type
from redisvl.utils.vectorize.base import BaseVectorizer, Vectorizers

logger = logging.getLogger(__name__)

class OllamaTextVectorizer(BaseVectorizer):
    """The OllamaTextVectorizer class utilizes Ollama's API to generate
    embeddings for text data using the nomic-embed-text model.

    This vectorizer is designed to interact with Ollama's embeddings API,
    requiring the `ollama` python client to be installed with
    `pip install ollama>=0.1.9`. The vectorizer supports both synchronous
    and asynchronous operations, allowing for batch processing of texts
    and flexibility in handling preprocessing tasks.

    You can optionally enable caching to improve performance when generating
    embeddings for repeated text inputs.

    .. code-block:: python
        vectorizer = OllamaTextVectorizer(
            model="nomic-embed-text"
        )
        embedding = vectorizer.embed("Hello, world!")
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model: str = "nomic-embed-text",
        api_config: Optional[Dict] = None,
        dtype: str = "float32",
        cache: Optional["EmbeddingsCache"] = None,
        **kwargs,
    ):
        """Initialize the Ollama vectorizer.

        Args:
            model (str): Model to use for embedding. Defaults to 'nomic-embed-text'.
            api_config (Optional[Dict], optional): Dictionary containing API configuration options.
            dtype (str): The default datatype for embeddings when returned as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                Defaults to 'float32'.
            cache (Optional[EmbeddingsCache]): Optional EmbeddingsCache instance to cache embeddings
                for better performance with repeated texts. Defaults to None.

        Raises:
            ImportError: If the ollama library is not installed.
            ValueError: If the Ollama API call fails or returns invalid data.
        """
        super().__init__(model=model, dtype=dtype, cache=cache)
        self._setup(api_config, **kwargs)

    def _setup(self, api_config: Optional[Dict], **kwargs):
        """Set up the Ollama client and determine the embedding dimensions."""
        self._initialize_client(api_config, **kwargs)
        self.dims = self._set_model_dims()

    def _initialize_client(self, api_config: Optional[Dict], **kwargs):
        """Setup the Ollama client.

        Args:
            api_config: Dictionary with API configuration options
            **kwargs: Additional arguments to pass to the Ollama client

        Raises:
            ImportError: If the ollama library is not installed
        """
        if api_config is None:
            api_config = {}
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "Ollama vectorizer requires the ollama library. "
                "Please install with `pip install ollama>=0.1.9`"
            )
        self._client = ollama.Client(**api_config, **kwargs)

    def _set_model_dims(self) -> int:
        """Determine the dimensionality of the embedding model by making a test call.

        Returns:
            int: Dimensionality of the embedding model

        Raises:
            ValueError: If embedding dimensions cannot be determined
        """
        try:
            embedding = self._embed("dimension check")
            return len(embedding)
        except Exception as e:
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def _embed(self, text: str, **kwargs) -> List[float]:
        """Generate a vector embedding for a single text using the Ollama API.

        Args:
            text: Text to embed
            **kwargs: Additional parameters to pass to the Ollama API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If text is not a string
            ValueError: If embedding fails
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")
        try:
            result = self._client.embeddings(model=self.model, prompt=text, **kwargs)
            return result["embedding"]
        except Exception as e:
            raise ValueError(f"Embedding text failed: {e}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def _embed_many(
        self, texts: List[str], batch_size: int = 10, **kwargs
    ) -> List[List[float]]:
        """Generate vector embeddings for a batch of texts using the Ollama API.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the Ollama API

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If texts is not a list of strings
            ValueError: If embedding fails
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if texts and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")
        embeddings: List = []
        for batch in self.batchify(texts, batch_size):
            try:
                for text in batch:
                    result = self._client.embeddings(model=self.model, prompt=text, **kwargs)
                    embeddings.append(result["embedding"])
            except Exception as e:
                raise ValueError(f"Embedding texts failed: {e}")
        return embeddings

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    async def _aembed(self, text: str, **kwargs) -> List[float]:
        """Asynchronously generate a vector embedding for a single text using the Ollama API.

        Args:
            text: Text to embed
            **kwargs: Additional parameters to pass to the Ollama API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If text is not a string
            ValueError: If embedding fails
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")
        try:
            import ollama
            result = await ollama.AsyncClient().embeddings(model=self.model, prompt=text, **kwargs)
            return result["embedding"]
        except Exception as e:
            raise ValueError(f"Embedding text failed: {e}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    async def _aembed_many(
        self, texts: List[str], batch_size: int = 10, **kwargs
    ) -> List[List[float]]:
        """Asynchronously generate vector embeddings for a batch of texts using the Ollama API.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the Ollama API

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If texts is not a list of strings
            ValueError: If embedding fails
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if texts and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")
        embeddings: List = []
        for batch in self.batchify(texts, batch_size):
            try:
                import ollama
                for text in batch:
                    result = await ollama.AsyncClient().embeddings(model=self.model, prompt=text, **kwargs)
                    embeddings.append(result["embedding"])
            except Exception as e:
                raise ValueError(f"Embedding texts failed: {e}")
        return embeddings

    @property
    def type(self) -> str:
        return Vectorizers.ollama.value