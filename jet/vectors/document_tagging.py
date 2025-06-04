from jet.file.utils import load_file
from jet.logger.config import colorize_log
from jet.token.token_utils import split_headers
from jet.vectors.document_types import HeaderDocument
from tqdm import tqdm
import string
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT

nltk.download('punkt')
nltk.download('stopwords')

# Load model
model_name = 'intfloat/e5-base-v2'
model = SentenceTransformer(model_name)
kw_model = KeyBERT(model_name)


def preprocess_texts(scraped_data):
    return [f"passage: {item['text']}" for item in scraped_data]


def extract_keywords(texts, top_n=3):
    stop_words = list(set(stopwords.words('english')
                          ).union(set(string.punctuation)))
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=100)
    tfidf_matrix = vectorizer.fit_transform(
        tqdm(texts, desc="Extracting keywords"))
    feature_names = vectorizer.get_feature_names_out()
    keywords = []
    for doc_idx in range(tfidf_matrix.shape[0]):
        tfidf_scores = tfidf_matrix[doc_idx].toarray()[0]
        top_indices = tfidf_scores.argsort()[-top_n:][::-1]
        keywords.append([feature_names[i] for i in top_indices])
    return keywords


def tag_documents(query, scraped_data, threshold="auto"):
    query_text = f"query: {query}"
    passages = preprocess_texts(scraped_data)

    query_embedding = model.encode(
        [query_text], normalize_embeddings=True, show_progress_bar=False)[0]
    passage_embeddings = model.encode(
        passages, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

    similarities = np.dot(passage_embeddings, query_embedding)

    if threshold == "auto":
        threshold = float(np.mean(similarities) + 0.5 * np.std(similarities))

    tags = []
    for i, sim in enumerate(tqdm(similarities, desc="Assigning relevance tags")):
        tag = "relevant" if sim >= threshold else "irrelevant"
        tags.append({
            "header": scraped_data[i]["header"],
            "text": scraped_data[i]["text"],
            "relevance_tag": tag,
            "relevance_similarity": float(sim)
        })
    return tags


def dynamic_label_tagging(scraped_data, num_clusters=2, threshold=0.8):
    passages = preprocess_texts(scraped_data)
    embeddings = model.encode(
        passages, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

    print("Clustering embeddings...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    with tqdm(total=1, desc="Performing K-means clustering") as pbar:
        cluster_labels = kmeans.fit_predict(embeddings)
        pbar.update(1)

    clusters = {i: [] for i in range(num_clusters)}
    for i, label in enumerate(cluster_labels):
        clusters[label].append(scraped_data[i]["text"])

    cluster_label_names = {}
    for cluster_id, texts in tqdm(clusters.items(), desc="Generating cluster labels"):
        if texts:
            joined = " ".join(texts)
            keywords = kw_model.extract_keywords(
                joined, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=2)
            cluster_label_names[cluster_id] = " ".join(
                [kw[0] for kw in keywords])[:30]
        else:
            cluster_label_names[cluster_id] = f"cluster_{cluster_id}"

    tagged_results = []
    for i, (data, cluster_id) in enumerate(tqdm(zip(scraped_data, cluster_labels), total=len(scraped_data), desc="Assigning topic labels")):
        sim_to_centroid = np.dot(
            embeddings[i], kmeans.cluster_centers_[cluster_id])
        labels = [{"label": cluster_label_names[cluster_id],
                   "similarity": float(sim_to_centroid)}]

        for other_cluster_id in range(num_clusters):
            if other_cluster_id != cluster_id:
                sim = np.dot(embeddings[i], kmeans.cluster_centers_[
                             other_cluster_id])
                if sim >= threshold:
                    labels.append(
                        {"label": cluster_label_names[other_cluster_id], "similarity": float(sim)})

        tagged_results.append({
            "header": data["header"],
            "text": data["text"],
            "topic_labels": labels
        })

    return tagged_results


def integrated_document_tagging(query, scraped_data, relevance_threshold="auto", topic_threshold=0.8, num_clusters=2):
    relevance_results = tag_documents(query, scraped_data, relevance_threshold)
    topic_results = dynamic_label_tagging(
        scraped_data, num_clusters, topic_threshold)

    integrated_results = []
    for rel, top in tqdm(zip(relevance_results, topic_results), total=len(scraped_data), desc="Combining tagging results"):
        assert rel["header"] == top["header"] and rel["text"] == top["text"], "Mismatch in results"
        integrated_results.append({
            "header": rel["header"],
            "text": rel["text"],
            "relevance_tag": rel["relevance_tag"],
            "relevance_similarity": rel["relevance_similarity"],
            "topic_labels": top["topic_labels"]
        })

    return integrated_results
