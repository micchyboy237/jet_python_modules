from bertopic import BERTopic
# from jet.logger import logger
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from umap import UMAP
import os
import pandas as pd
import re
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def example_visualize_topics():
    print("""Example to visualize topics in a 2D representation.""")
    # Given: A set of documents from the 20 newsgroups dataset
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = newsgroups.data[:1000]
   
    # When: We train a BERTopic model and visualize topics
    # Use a safer UMAP for internal dimensionality reduction (n_neighbors=15 avoids small-k issues)
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,  # BERTopic default; keeps it generic
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        init='random'  # Bypass spectral for robustness on small N
    )
    topic_model = BERTopic(umap_model=umap_model, verbose=True)
    topics, probs = topic_model.fit_transform(docs)
    
    # Override viz UMAP with safe params (n_neighbors=15, random init) to fix spectral error
    viz_umap = UMAP(
        n_neighbors=15,
        n_components=2,
        metric='cosine',
        random_state=42,
        init='random'  # Avoids eigsh fallback entirely
    )
    fig = topic_model.visualize_topics(umap_model=viz_umap)
   
    # Then: Save the interactive plot to HTML
    output_path = os.path.join(OUTPUT_DIR, "viz.html")
    fig.write_html(output_path)
    print(f"Topic visualization saved to {output_path}")

def example_visualize_documents():
    print("""Example to visualize documents with reduced embeddings.""")
    # Given: Documents and a pre-trained sentence transformer model
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = newsgroups.data[:1000]
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)
    
    # When: We train BERTopic and visualize documents with reduced embeddings
    topic_model = BERTopic(verbose=True).fit(docs, embeddings)
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    fig = topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings, hide_document_hover=True)
    
    # Then: Save the document visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "documents.html")
    fig.write_html(output_path)
    print(f"Document visualization saved to {output_path}")

def example_visualize_documents_with_titles():
    print("""Example to visualize documents with custom titles.""")
    # Given: Documents with generated titles
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = newsgroups.data[:1000]
    titles = [f"Doc {i+1}: {doc[:50]}..." for i, doc in enumerate(docs)]
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)
    
    # When: We train BERTopic and visualize documents with titles
    topic_model = BERTopic(verbose=True).fit(docs, embeddings)
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    fig = topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings, hide_document_hover=False)
    
    # Then: Save the visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "documents_with_titles.html")
    fig.write_html(output_path)
    print(f"Document visualization with titles saved to {output_path}")

def example_visualize_hierarchy():
    print("""Example to visualize topic hierarchy.""")
    # Given: Documents for topic modeling
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = newsgroups.data[:1000]
    
    # When: We train BERTopic and create a hierarchical visualization
    topic_model = BERTopic(verbose=True)
    topics, probs = topic_model.fit_transform(docs)
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    
    # Then: Save the hierarchy visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "hierarchy.html")
    fig.write_html(output_path)
    print(f"Hierarchy visualization saved to {output_path}")

def example_visualize_hierarchical_documents():
    print("""Example to visualize hierarchical documents.""")
    # Given: Documents and embeddings
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = newsgroups.data[:1000]
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)
    
    # When: We train BERTopic and visualize hierarchical documents
    topic_model = BERTopic(verbose=True).fit(docs, embeddings)
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    fig = topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, reduced_embeddings=reduced_embeddings, hide_document_hover=True)
    
    # Then: Save the visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "hierarchical_documents.html")
    fig.write_html(output_path)
    print(f"Hierarchical documents visualization saved to {output_path}")

def example_visualize_barchart():
    print("""Example to visualize topic term bar charts.""")
    # Given: Documents for topic modeling
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = newsgroups.data[:1000]
    
    # When: We train BERTopic and create a bar chart visualization
    topic_model = BERTopic(verbose=True)
    topics, probs = topic_model.fit_transform(docs)
    fig = topic_model.visualize_barchart()
    
    # Then: Save the bar chart to HTML
    output_path = os.path.join(OUTPUT_DIR, "bar_chart.html")
    fig.write_html(output_path)
    print(f"Bar chart visualization saved to {output_path}")

def example_visualize_heatmap():
    print("""Example to visualize topic similarity heatmap.""")
    # Given: Documents for topic modeling
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = newsgroups.data[:1000]
    
    # When: We train BERTopic and create a heatmap visualization
    topic_model = BERTopic(verbose=True)
    topics, probs = topic_model.fit_transform(docs)
    fig = topic_model.visualize_heatmap(n_clusters=5)
    
    # Then: Save the heatmap to HTML
    output_path = os.path.join(OUTPUT_DIR, "heatmap.html")
    fig.write_html(output_path)
    print(f"Heatmap visualization saved to {output_path}")

def example_visualize_term_rank():
    print("""Example to visualize term rank decline.""")
    # Given: Documents for topic modeling
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = newsgroups.data[:1000]
    
    # When: We train BERTopic and visualize term rank decline
    topic_model = BERTopic(verbose=True)
    topics, probs = topic_model.fit_transform(docs)
    fig = topic_model.visualize_term_rank(log_scale=True)
    
    # Then: Save the term rank visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "term_rank_log.html")
    fig.write_html(output_path)
    print(f"Term rank visualization saved to {output_path}")

def example_visualize_topics_over_time():
    print("""Example to visualize topics over time.""")
    # Given: Tweets with timestamps
    trump = pd.read_csv('https://drive.google.com/uc?export=download&id=1xRKHaP-QwACMydlDnyFPEaFdtskJuBa6')
    trump.text = trump.apply(lambda row: re.sub(r"http\S+", "", row.text).lower(), 1)
    trump.text = trump.apply(lambda row: " ".join(filter(lambda x:x[0]!="@", row.text.split())), 1)
    trump.text = trump.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.text).split()), 1)
    trump = trump.loc[(trump.isRetweet == "f") & (trump.text != ""), :]
    timestamps = trump.date.to_list()
    tweets = trump.text.to_list()
    
    # When: We train BERTopic and visualize topics over time
    model = BERTopic(verbose=True)
    topics, probs = model.fit_transform(tweets)
    topics_over_time = model.topics_over_time(tweets, timestamps)
    fig = model.visualize_topics_over_time(topics_over_time, topics=[9, 10, 72, 83, 87, 91])
    
    # Then: Save the visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "trump.html")
    fig.write_html(output_path)
    print(f"Topics over time visualization saved to {output_path}")

def example_visualize_topics_per_class():
    print("""Example to visualize topics per class.""")
    # Given: Documents with class labels
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = data["data"]
    classes = [data["target_names"][i] for i in data["target"]]
    
    # When: We train BERTopic and visualize topics per class
    topic_model = BERTopic(verbose=True)
    topics, probs = topic_model.fit_transform(docs)
    topics_per_class = topic_model.topics_per_class(docs, classes=classes)
    fig = topic_model.visualize_topics_per_class(topics_per_class)
    
    # Then: Save the visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "topics_per_class.html")
    fig.write_html(output_path)
    print(f"Topics per class visualization saved to {output_path}")

def example_visualize_distribution():
    print("""Example to visualize topic probability distribution.""")
    # Given: Documents for topic modeling
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = newsgroups.data[:1000]
    
    # When: We train BERTopic with probabilities and visualize distribution
    topic_model = BERTopic(calculate_probabilities=True)
    topics, probs = topic_model.fit_transform(docs)
    fig = topic_model.visualize_distribution(probs[0])
    
    # Then: Save the distribution visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "probabilities.html")
    fig.write_html(output_path)
    print(f"Distribution visualization saved to {output_path}")

def example_visualize_approximate_distribution():
    print("""Example to visualize approximate token-level distribution.""")
    # Given: Documents for topic modeling
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = newsgroups.data[:1000]
    
    # When: We train BERTopic and calculate token-level distributions
    topic_model = BERTopic(verbose=True)
    topics, _ = topic_model.fit_transform(docs)
    topic_distr, topic_token_distr = topic_model.approximate_distribution(docs, calculate_tokens=True)
    df = topic_model.visualize_approximate_distribution(docs[1], topic_token_distr[1])
    
    # Then: Save the dataframe to HTML
    output_path = os.path.join(OUTPUT_DIR, "approximate_distribution.html")
    df.to_html(output_path)
    print(f"Approximate distribution visualization saved to {output_path}")

if __name__ == "__main__":
    example_visualize_topics()
    example_visualize_documents()
    example_visualize_documents_with_titles()
    example_visualize_hierarchy()
    example_visualize_hierarchical_documents()
    example_visualize_barchart()
    example_visualize_heatmap()
    example_visualize_term_rank()
    example_visualize_topics_over_time()
    example_visualize_topics_per_class()
    example_visualize_distribution()
    example_visualize_approximate_distribution()
    print("\n\n[DONE]", bright=True)