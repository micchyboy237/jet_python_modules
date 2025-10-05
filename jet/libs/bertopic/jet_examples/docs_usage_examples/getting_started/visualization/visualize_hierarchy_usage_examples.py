from bertopic import BERTopic
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

def example_visualize_hierarchy():
    """Example to visualize topic hierarchy."""
    # Given: Documents for topic modeling
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    
    # When: We train BERTopic and create a hierarchical visualization
    topic_model = BERTopic(verbose=True)
    topics, probs = topic_model.fit_transform(docs)
    fig = topic_model.visualize_hierarchy()
    
    # Then: Save the hierarchy visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "hierarchy.html")
    fig.write_html(output_path)
    logger.info(f"Hierarchy visualization saved to {output_path}")

def example_visualize_hierarchy_with_topics():
    """Example to visualize hierarchy with calculated hierarchical topics."""
    # Given: Documents for topic modeling
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    
    # When: We train BERTopic and visualize hierarchical topics
    topic_model = BERTopic(verbose=True)
    topics, probs = topic_model.fit_transform(docs)
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    
    # Then: Save the hierarchy visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "hierarchical_topics.html")
    fig.write_html(output_path)
    logger.info(f"Hierarchical topics visualization saved to {output_path}")

def example_get_topic_tree():
    """Example to generate and log topic tree hierarchy."""
    # Given: Documents for topic modeling
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    
    # When: We train BERTopic and generate topic tree
    topic_model = BERTopic(verbose=True)
    topics, probs = topic_model.fit_transform(docs)
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    tree = topic_model.get_topic_tree(hierarchical_topics)
    
    # Then: Log the topic tree
    output_path = os.path.join(OUTPUT_DIR, "topic_tree.txt")
    with open(output_path, 'w') as f:
        f.write(tree)
    logger.info(f"Topic tree saved to {output_path}")
    logger.debug(tree)

def example_visualize_hierarchical_documents():
    """Example to visualize hierarchical documents."""
    # Given: Documents and embeddings
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)
    
    # When: We train BERTopic and visualize hierarchical documents
    topic_model = BERTopic().fit(docs, embeddings)
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    fig = topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, reduced_embeddings=reduced_embeddings, hide_document_hover=True)
    
    # Then: Save the visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "hierarchical_documents.html")
    fig.write_html(output_path)
    logger.info(f"Hierarchical documents visualization saved to {output_path}")

if __name__ == "__main__":
    example_visualize_hierarchy()
    example_visualize_hierarchy_with_topics()
    example_get_topic_tree()
    example_visualize_hierarchical_documents()
    logger.info("\n\n[DONE]", bright=True)