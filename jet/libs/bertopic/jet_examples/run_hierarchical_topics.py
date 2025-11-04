from typing import List, Literal
from scipy.cluster import hierarchy as sch
from jet.adapters.bertopic import BERTopic
# from sklearn.datasets import fetch_20newsgroups
from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_data
import os
import shutil

from jet.logger import logger

OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def generate_hierarchical_topics(
    docs: List[str],
    method: Literal["ward", "single", "complete", "average", "centroid", "median"] = "ward"
):
    logger.info(f"Generating '{method}' hierarchical topics...")
    output_dir = f"{OUTPUT_DIR}/{method}"

    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)

    document_info = topic_model.get_document_info(docs)
    topic_info = topic_model.get_topic_info()

    linkage_function = lambda x: sch.linkage(x, method, optimal_ordering=True)

    hierarchical_topics = topic_model.hierarchical_topics(
        docs,
        linkage_function=linkage_function
    )
    tree = topic_model.get_topic_tree(hierarchical_topics)

    logger.debug(f"Method: '{method}' hierarchical tree:")
    logger.success(tree)

    
    save_file(document_info.to_dict(orient="records"), f"{output_dir}/document_info.json")
    save_file(topic_info.to_dict(orient="records"), f"{output_dir}/topic_info.json")
    save_file(hierarchical_topics.to_dict(orient="records"), f"{output_dir}/hierarchical_topics.json")
    save_file(topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics).to_image(), f"{output_dir}/plots.png")
    save_file(tree, f"{output_dir}/tree.txt")

# Example Usage
if __name__ == "__main__":
    docs = load_sample_data()
    save_file(docs, f"{OUTPUT_DIR}/docs.json")
    # docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

    for method in ["ward", "single", "complete", "average", "centroid", "median"]:
        generate_hierarchical_topics(docs, method)