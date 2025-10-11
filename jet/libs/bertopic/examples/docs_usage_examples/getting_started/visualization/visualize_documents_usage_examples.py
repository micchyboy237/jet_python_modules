from jet.adapters.bertopic import BERTopic
from jet.logger import logger
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from umap import UMAP
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

def example_visualize_documents():
    """Example to visualize documents with reduced embeddings."""
    # Given: Documents from the 20 newsgroups dataset
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)
    
    # When: We train BERTopic and visualize documents
    topic_model = BERTopic().fit(docs, embeddings)
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    fig = topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings, hide_document_hover=True)
    
    # Then: Save the document visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "documents.html")
    fig.write_html(output_path)
    logger.info(f"Document visualization saved to {output_path}")

def example_visualize_documents_with_titles():
    """Example to visualize documents with custom titles."""
    # Given: Documents with generated titles
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    titles = [f"Doc {i+1}: {doc[:50]}..." for i, doc in enumerate(docs)]
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)
    
    # When: We train BERTopic and visualize documents with titles
    topic_model = BERTopic().fit(docs, embeddings)
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    fig = topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings, hide_document_hover=False)
    
    # Then: Save the visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "documents_with_titles.html")
    fig.write_html(output_path)
    logger.info(f"Document visualization with titles saved to {output_path}")

def example_visualize_document_datamap():
    """Example to visualize documents using DataMapPlot."""
    # Given: Documents and embeddings
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)
    
    # When: We train BERTopic and create a DataMapPlot
    topic_model = BERTopic().fit(docs, embeddings)
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    fig = topic_model.visualize_document_datamap(docs, reduced_embeddings=reduced_embeddings, interactive=True)
    
    # Then: Save the DataMapPlot to HTML
    output_path = os.path.join(OUTPUT_DIR, "datamapplot.html")
    fig.write_html(output_path)
    logger.info(f"DataMapPlot visualization saved to {output_path}")

def example_visualize_document_datamap_save_png():
    """Example to save DataMapPlot as PNG."""
    # Given: Documents and embeddings
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)
    
    # When: We train BERTopic and create a DataMapPlot
    topic_model = BERTopic().fit(docs, embeddings)
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    fig = topic_model.visualize_document_datamap(docs, reduced_embeddings=reduced_embeddings)
    
    # Then: Save the DataMapPlot as PNG
    output_path = os.path.join(OUTPUT_DIR, "datamapplot.png")
    fig.savefig(output_path, bbox_inches="tight")
    logger.info(f"DataMapPlot saved to {output_path}")

def example_visualize_distribution():
    """Example to visualize topic probability distribution."""
    # Given: Documents for topic modeling
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    
    # When: We train BERTopic with probabilities and visualize distribution
    topic_model = BERTopic(calculate_probabilities=True)
    topics, probs = topic_model.fit_transform(docs)
    fig = topic_model.visualize_distribution(probs[0])
    
    # Then: Save the distribution visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "probabilities.html")
    fig.write_html(output_path)
    logger.info(f"Distribution visualization saved to {output_path}")

def example_visualize_approximate_distribution():
    """Example to visualize approximate token-level distribution."""
    # Given: Documents for topic modeling
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    
    # When: We train BERTopic and calculate token-level distributions
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(docs)
    topic_distr, topic_token_distr = topic_model.approximate_distribution(docs, calculate_tokens=True)
    df = topic_model.visualize_approximate_distribution(docs[1], topic_token_distr[1])
    
    # Then: Save the dataframe to HTML
    output_path = os.path.join(OUTPUT_DIR, "approximate_distribution.html")
    df.to_html(output_path)
    logger.info(f"Approximate distribution visualization saved to {output_path}")

if __name__ == "__main__":
    example_visualize_documents()
    example_visualize_documents_with_titles()
    example_visualize_document_datamap()
    example_visualize_document_datamap_save_png()
    example_visualize_distribution()
    example_visualize_approximate_distribution()
    logger.info("\n\n[DONE]", bright=True)