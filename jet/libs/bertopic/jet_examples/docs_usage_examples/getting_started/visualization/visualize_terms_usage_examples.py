from bertopic import BERTopic
from jet.logger import logger
from sklearn.datasets import fetch_20newsgroups
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

def example_visualize_barchart():
    """Example to visualize topic term bar charts."""
    # Given: Documents for topic modeling
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    
    # When: We train BERTopic and create a bar chart visualization
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
    fig = topic_model.visualize_barchart()
    
    # Then: Save the bar chart to HTML
    output_path = os.path.join(OUTPUT_DIR, "bar_chart.html")
    fig.write_html(output_path)
    logger.info(f"Bar chart visualization saved to {output_path}")

def example_visualize_term_rank():
    """Example to visualize term rank decline."""
    # Given: Documents for topic modeling
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    
    # When: We train BERTopic and visualize term rank decline
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
    fig = topic_model.visualize_term_rank()
    
    # Then: Save the term rank visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "term_rank.html")
    fig.write_html(output_path)
    logger.info(f"Term rank visualization saved to {output_path}")

def example_visualize_term_rank_log_scale():
    """Example to visualize term rank decline with log scale."""
    # Given: Documents for topic modeling
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    
    # When: We train BERTopic and visualize term rank decline with log scale
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
    fig = topic_model.visualize_term_rank(log_scale=True)
    
    # Then: Save the term rank visualization to HTML
    output_path = os.path.join(OUTPUT_DIR, "term_rank_log.html")
    fig.write_html(output_path)
    logger.info(f"Term rank visualization with log scale saved to {output_path}")

if __name__ == "__main__":
    example_visualize_barchart()
    example_visualize_term_rank()
    example_visualize_term_rank_log_scale()
    logger.info("\n\n[DONE]", bright=True)