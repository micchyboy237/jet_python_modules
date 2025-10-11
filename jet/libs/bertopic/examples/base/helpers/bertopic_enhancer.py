import os
import glob
import zipfile
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Optional
from tqdm import tqdm
from PIL import Image
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from umap import UMAP
from hdbscan import HDBSCAN
from wordcloud import WordCloud
import matplotlib.pyplot as plt
try:
    from model2vec import StaticModel
    MODEL2VEC_AVAILABLE = True
except ImportError:
    MODEL2VEC_AVAILABLE = False
try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
from datasets import load_dataset

class BERTopicEnhancer:
    """A utility class for enhancing BERTopic with various tips and tricks."""
    
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        vectorizer_model: Optional[CountVectorizer] = None,
        ctfidf_model: Optional[ClassTfidfTransformer] = None,
        representation_model: Optional[Any] = None,
        umap_model: Optional[Any] = None,
        hdbscan_model: Optional[Any] = None
    ):
        """Initialize BERTopic with optional custom models."""
        self.embedding_model = embedding_model or SentenceTransformer("all-MiniLM-L6-v2")
        self.vectorizer_model = vectorizer_model or CountVectorizer(stop_words="english")
        self.ctfidf_model = ctfidf_model
        self.representation_model = representation_model
        self.umap_model = umap_model
        self.hdbscan_model = hdbscan_model
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model
        )

    def remove_stopwords_countvectorizer(self) -> None:
        """Configure BERTopic to remove stopwords using CountVectorizer."""
        self.vectorizer_model = CountVectorizer(stop_words="english")
        self.topic_model.vectorizer_model = self.vectorizer_model

    def reduce_frequent_words_ctfidf(self) -> None:
        """Configure BERTopic to reduce frequent words using ClassTfidfTransformer."""
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        self.topic_model.ctfidf_model = self.ctfidf_model

    def keybert_inspired_representation(self) -> None:
        """Configure BERTopic with KeyBERT-inspired representation."""
        self.representation_model = KeyBERTInspired()
        self.topic_model.representation_model = self.representation_model

    def diversify_topics_mmr(self, diversity: float = 0.2) -> None:
        """Configure BERTopic with Maximal Marginal Relevance for topic diversification."""
        self.representation_model = MaximalMarginalRelevance(diversity=diversity)
        self.topic_model.representation_model = self.representation_model

    def get_topic_term_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Extract the c-TF-IDF topic-term matrix and corresponding words."""
        return self.topic_model.c_tf_idf_, self.topic_model.vectorizer_model.get_feature_names_out()

    def precompute_embeddings(self, docs: List[str]) -> np.ndarray:
        """Pre-compute embeddings for documents."""
        return self.embedding_model.encode(docs, show_progress_bar=False)

    def speed_up_umap(self, embeddings: np.ndarray, n_components: int = 5) -> None:
        """Configure UMAP with rescaled PCA embeddings for faster processing."""
        def rescale(x, inplace=False):
            if not inplace:
                x = np.array(x, copy=True)
            x /= np.std(x[:, 0]) * 10000
            return x

        pca_embeddings = rescale(PCA(n_components=n_components).fit_transform(embeddings))
        self.umap_model = UMAP(
            n_neighbors=15,
            n_components=n_components,
            min_dist=0.0,
            metric="cosine",
            init=pca_embeddings
        )
        self.topic_model.umap_model = self.umap_model

    def tune_hyperparameters(
        self,
        language: str = "english",
        top_n_words: int = 10,
        n_gram_range: Tuple[int, int] = (1, 2),
        min_topic_size: int = 10,
        nr_topics: Optional[Any] = None,
        low_memory: bool = False,
        calculate_probabilities: bool = False,
        umap_n_neighbors: int = 15,
        umap_n_components: int = 5,
        umap_metric: str = "cosine",
        umap_low_memory: bool = False,
        hdbscan_min_cluster_size: int = 10,
        hdbscan_min_samples: Optional[int] = None,
        hdbscan_metric: str = "euclidean"
    ) -> None:
        """Tune BERTopic, UMAP, and HDBSCAN hyperparameters."""
        if language == "multilingual":
            self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        else:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vectorizer_model = CountVectorizer(stop_words="english", ngram_range=n_gram_range)
        self.umap_model = UMAP(
            n_neighbors=umap_n_neighbors,
            n_components=umap_n_components,
            metric=umap_metric,
            low_memory=umap_low_memory
        )
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples or hdbscan_min_cluster_size,
            metric=hdbscan_metric,
            prediction_data=True
        )
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            top_n_words=top_n_words,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            low_memory=low_memory,
            calculate_probabilities=calculate_probabilities
        )

    def create_wordcloud(self, topic: int) -> None:
        """Generate and display a word cloud for a specific topic."""
        text = {word: value for word, value in self.topic_model.get_topic(topic)}
        wc = WordCloud(background_color="white", max_words=1000)
        wc.generate_from_frequencies(text)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def compare_topic_models(self, other_model: BERTopic) -> np.ndarray:
        """Compare topic embeddings between two BERTopic models using cosine similarity."""
        return cosine_similarity(self.topic_model.topic_embeddings_, other_model.topic_embeddings_)

    def process_multimodal_data(
        self,
        images: List[str],
        captions: List[str],
        batch_size: int = 32,
        embedding_model_name: str = "clip-ViT-B-32"
    ) -> Tuple[List[int], np.ndarray, pd.DataFrame]:
        """Process multimodal data (images and captions) and fit BERTopic.

        Args:
            images: List of image file paths.
            captions: List of corresponding captions.
            batch_size: Number of images to process per batch (default: 32).
            embedding_model_name: Name of the sentence-transformers model for image embeddings (default: clip-ViT-B-32).

        Returns:
            Tuple containing (topics, probabilities, processed DataFrame with topic assignments).

        Raises:
            FileNotFoundError: If image files or captions are inaccessible.
            ValueError: If images and captions lists have mismatched lengths.
        """
        if len(images) != len(captions):
            raise ValueError(f"Mismatch between images ({len(images)}) and captions ({len(captions)}) lengths")
        
        try:
            model = SentenceTransformer(embedding_model_name)
            nr_iterations = int(np.ceil(len(images) / batch_size))
            embeddings = []
            
            for i in tqdm(range(nr_iterations), desc="Embedding images"):
                start_index = i * batch_size
                end_index = min((i * batch_size) + batch_size, len(images))
                
                # Use context manager for image handling
                with contextlib.ExitStack() as stack:
                    images_to_embed = [
                        stack.enter_context(Image.open(filepath))
                        for filepath in images[start_index:end_index]
                        if os.path.exists(filepath)
                    ]
                    if not images_to_embed:
                        continue
                    img_emb = model.encode(images_to_embed, show_progress_bar=False)
                    embeddings.extend(img_emb.tolist())
            
            embeddings = np.array(embeddings)
            if len(embeddings) != len(captions):
                raise ValueError(f"Generated {len(embeddings)} embeddings but expected {len(captions)}")
            
            topics, probs = self.topic_model.fit_transform(captions, embeddings)
            df = pd.DataFrame({"img_id": images[:len(topics)], "img_caption": captions[:len(topics)], "Topic": topics})
            
            return topics, probs, df
        
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error accessing image files or captions: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to process multimodal data: {str(e)}")

    def keybert_vocabulary(self, docs: List[str]) -> None:
        """Use KeyBERT to create a vocabulary and configure BERTopic."""
        if not KEYBERT_AVAILABLE:
            raise ImportError("KeyBERT is not installed. Install it to use this method.")
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(docs)
        vocabulary = list(set(k[0] for keyword in keywords for k in keyword))
        self.vectorizer_model = CountVectorizer(vocabulary=vocabulary)
        self.topic_model.vectorizer_model = self.vectorizer_model

    def approximate_document_distribution(
        self,
        docs: List[str],
        batch_size: Optional[int] = None,
        window: Optional[int] = None,
        stride: Optional[int] = None,
        use_embedding_model: bool = False,
        calculate_tokens: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Approximate topic distributions for documents."""
        return self.topic_model.approximate_distribution(
            docs,
            batch_size=batch_size,
            window=window,
            stride=stride,
            use_embedding_model=use_embedding_model,
            calculate_tokens=calculate_tokens
        )

    def enable_lightweight_embedding(self, model_name: str = "minishlab/potion-base-8M") -> None:
        """Configure BERTopic with Model2Vec for lightweight, CPU-friendly embeddings."""
        if not MODEL2VEC_AVAILABLE:
            raise ImportError("model2vec is not installed. Install it to use lightweight embeddings: pip install model2vec")
        self.embedding_model = StaticModel.from_pretrained(model_name)
        self.topic_model.embedding_model = self.embedding_model

if __name__ == "__main__":
    import logging
    import contextlib
    logging.basicConfig(level=logging.DEBUG)  # Temporary debug logging
    logger = logging.getLogger(__name__)

    # Example usage with 20newsgroups dataset
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data'][:1000]  # Limited for demo
    enhancer = BERTopicEnhancer()

    # Option 1: Pre-compute embeddings with default model (sentence-transformers)
    embeddings = enhancer.precompute_embeddings(docs)

    # Option 2: Enable lightweight Model2Vec (uncomment to use instead of pre-computed embeddings)
    # enhancer.enable_lightweight_embedding()
    # embeddings = None  # Model2Vec will handle embedding computation internally

    # Tune hyperparameters for a medium-sized dataset
    enhancer.tune_hyperparameters(
        language="english",
        top_n_words=15,
        n_gram_range=(1, 2),
        min_topic_size=50,
        nr_topics="auto",
        low_memory=True,
        calculate_probabilities=True,
        umap_n_neighbors=20,
        umap_n_components=5,
        umap_metric="cosine",
        umap_low_memory=True,
        hdbscan_min_cluster_size=50,
        hdbscan_min_samples=25,
        hdbscan_metric="euclidean"
    )

    # Fit model
    topics, probs = enhancer.topic_model.fit_transform(docs, embeddings)

    # Approximate document distribution
    topic_distr, topic_token_distr = enhancer.approximate_document_distribution(docs, window=4, calculate_tokens=True)

    # Visualize results
    logger.debug(f"Topic distribution for first document: {topic_distr[0]}")
    logger.debug(f"Max probability in topic_distr[0]: {np.max(topic_distr[0])}")
    if np.max(topic_distr[0]) > 0.001:  # Check if probabilities are sufficient
        enhancer.topic_model.visualize_distribution(topic_distr[0], min_probability=0.001)
    else:
        logger.warning("Skipping visualization: All probabilities in topic_distr[0] are below 0.001")

    enhancer.create_wordcloud(topic=1)

    # Example comparing two models
    en_docs = load_dataset("stsb_multi_mt", name="en", split="train").to_pandas().sentence1.tolist()[:500]
    nl_docs = load_dataset("stsb_multi_mt", name="nl", split="train").to_pandas().sentence1.tolist()[:500]
    en_enhancer = BERTopicEnhancer(embedding_model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"))
    nl_enhancer = BERTopicEnhancer(embedding_model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"))
    en_enhancer.tune_hyperparameters(language="multilingual", min_topic_size=30, hdbscan_min_cluster_size=30)
    nl_enhancer.tune_hyperparameters(language="multilingual", min_topic_size=30, hdbscan_min_cluster_size=30)
    en_enhancer.topic_model.fit(en_docs)
    nl_enhancer.topic_model.fit(nl_docs)
    
    # Debug: Log topic counts
    en_topics = en_enhancer.topic_model.get_topic_info()
    nl_topics = nl_enhancer.topic_model.get_topic_info()
    logger.debug(f"English model topics: {len(en_topics)} (including -1 outlier)")
    logger.debug(f"Dutch model topics: {len(nl_topics)} (including -1 outlier)")

    sim_matrix = en_enhancer.compare_topic_models(nl_enhancer.topic_model)
    logger.debug(f"Similarity matrix shape: {sim_matrix.shape}")

    # Select a valid topic index (first non-outlier topic)
    if len(en_topics) > 1:  # Ensure at least one non-outlier topic exists
        topic_idx = 1  # Topic 0 (index 1 due to -1 offset)
        most_similar_topic = np.argmax(sim_matrix[topic_idx]) - 1
        print(f"Most similar Dutch topic for English topic 0: {most_similar_topic}")
    else:
        print("Not enough topics in English model to compare (only outlier topic found)")

    sys.exit()
    # Example multimodal data processing (Flickr 8k dataset)
    try:
        # Prepare Flickr 8k dataset
        img_folder = 'photos/'
        caps_folder = 'captions/'
        if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
            os.makedirs(img_folder, exist_ok=True)
            os.makedirs(caps_folder, exist_ok=True)
            if not os.path.exists('Flickr8k_Dataset.zip'):
                util.http_get(
                    'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
                    'Flickr8k_Dataset.zip'
                )
                util.http_get(
                    'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip',
                    'Flickr8k_text.zip'
                )
            for folder, file in [(img_folder, 'Flickr8k_Dataset.zip'), (caps_folder, 'Flickr8k_text.zip')]:
                with zipfile.ZipFile(file, 'r') as zf:
                    for member in tqdm(zf.infolist(), desc='Extracting'):
                        zf.extract(member, folder)

        images = list(glob.glob('photos/Flicker8k_Dataset/*.jpg'))
        captions = pd.read_csv(
            "captions/Flickr8k.lemma.token.txt", sep='\t', names=["img_id", "img_caption"]
        )
        captions.img_id = captions.apply(
            lambda row: "photos/Flicker8k_Dataset/" + row.img_id.split(".jpg")[0] + ".jpg", axis=1
        )
        captions = captions.groupby(["img_id"])["img_caption"].apply(','.join).reset_index()
        captions = pd.merge(captions, pd.Series(images, name="img_id"), on="img_id")
        images = captions.img_id.to_list()[:100]  # Limited for demo
        docs = captions.img_caption.to_list()[:100]

        # Process multimodal data
        topics, probs, df = enhancer.process_multimodal_data(images, docs, batch_size=16)
        print(f"Sample topic assignments:\n{df.head()}")
        enhancer.create_wordcloud(topic=2)  # Example: Visualize topic about skateboarders

    except FileNotFoundError as e:
        print(f"Multimodal data processing skipped: {str(e)}")
    except Exception as e:
        print(f"Error in multimodal data processing: {str(e)}")