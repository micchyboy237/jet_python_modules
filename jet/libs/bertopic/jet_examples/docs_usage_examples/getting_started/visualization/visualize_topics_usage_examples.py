from jet.adapters.bertopic import BERTopic
from jet.logger import logger
from sklearn.datasets import fetch_20newsgroups
import os
import pandas as pd
import re
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

def example_visualize_topics():
    """Example to visualize topics in a 2D representation."""
    # Given: Documents from the 20 newsgroups dataset
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    
    # When: We train BERTopic and visualize topics
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
    fig = topic_model.visualize_topics()
    
    # Then: Save the interactive plot to HTML
    output_path = os.path.join(OUTPUT_DIR, "viz.html")
    fig.write_html(output_path)
    logger.info(f"Topic visualization saved to {output_path}")

def example_visualize_heatmap():
    """Example to visualize topic similarity heatmap."""
    # Given: Documents for topic modeling
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    
    # When: We train BERTopic and create a heatmap visualization
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
    fig = topic_model.visualize_heatmap(n_clusters=5)
    
    # Then: Save the heatmap to HTML
    output_path = os.path.join(OUTPUT_DIR, "heatmap.html")
    fig.write_html(output_path)
    logger.info(f"Heatmap visualization saved to {output_path}")

def example_visualize_topics_over_time():
    """Example to visualize topics over time."""
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
    logger.info(f"Topics over time visualization saved to {output_path}")

def example_visualize_topics_per_class():
    """Example to visualize topics per class."""
    # Given: Documents with class labels
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = data["data"]
    classes = [data["target_names"][i] for i in data["target"]]
    
    # When: We train BERTopic and visualize topics per class
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
    topics_per_class = topic_model.topics_per_class(docs, classes=classes)
    fig = topic_model.visualize_topics_per_class(topics_per_class)
    
    # Then: Save the visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "topics_per_class.html")
    fig.write_html(output_path)
    logger.info(f"Topics per class visualization saved to {output_path}")

if __name__ == "__main__":
    example_visualize_topics()
    example_visualize_heatmap()
    example_visualize_topics_over_time()
    example_visualize_topics_per_class()
    logger.info("\n\n[DONE]", bright=True)