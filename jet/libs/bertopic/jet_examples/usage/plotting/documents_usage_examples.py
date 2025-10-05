import logging
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic
from umap import UMAP
import os
import shutil
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_HOME = os.path.expanduser("~/scikit_learn_data")

def load_sample_data():
    """Load sample dataset from 20 newsgroups."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', shuffle=False, remove=('headers', 'footers', 'quotes'), data_home=DATA_HOME, download_if_missing=True)
    documents = newsgroups.data[:1000]
    return documents

def example_visualize_documents():
    """Demonstrate visualizing documents in a BERTopic model."""
    logging.info("Starting visualize documents example...")
    documents = load_sample_data()
    topic_model = BERTopic(verbose=True)
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Generating reduced embeddings...")
    embeddings = topic_model._extract_embeddings(documents)
    umap_model = UMAP(n_components=2, random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings)
    
    logging.info("Visualizing documents...")
    fig = topic_model.visualize_documents(documents, embeddings=reduced_embeddings, hide_document_hover=True)
    fig.write_html(f"{OUTPUT_DIR}/fig.html")
    fig.write_image(f"{OUTPUT_DIR}/fig.png")
    save_file(fig.to_json(), f"{OUTPUT_DIR}/fig_raw_data.json")
    save_file(fig.to_dict(), f"{OUTPUT_DIR}/fig_data.json")
    fig_topics = [int(data["name"].split("_")[0]) for data in fig.to_dict()["data"][1:]]
    logging.info(f"Visualized topics: {fig_topics}")
    
    return fig

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_visualize_documents()
    logging.info("Documents usage examples completed successfully.")
