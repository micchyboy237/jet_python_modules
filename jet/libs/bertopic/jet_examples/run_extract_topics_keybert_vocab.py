
from typing import List, TypedDict
from jet.file.utils import save_file
import os
import shutil

from jet.logger import logger

OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

class TopicCategory(TypedDict):
    label: int
    category: str
    representation: List[str]
    count: int

class TopicEntry(TopicCategory):
    doc_index: int
    text: str  # Original chunk text

# Example Usage
if __name__ == "__main__":
    # Create vocabulary using KeyBERT

    from jet.adapters.keybert import KeyBERT
    from jet.libs.bertopic.examples.mock import load_sample_data

    logger.info("We first need to run KeyBERT on our data and create our vocabulary")
    # Prepare documents
    # docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    docs = load_sample_data()
    save_file(docs, f"{OUTPUT_DIR}/docs.json")

    # Extract keywords
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(docs)
    save_file(keywords, f"{OUTPUT_DIR}/keywords.json")

    # Create our vocabulary
    vocabulary = [k[0] for keyword in keywords for k in keyword]
    vocabulary = list(set(vocabulary))
    save_file(vocabulary, f"{OUTPUT_DIR}/vocabulary.json")

    
    # Pass vocabulary to BERTopic

    from jet.adapters.bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    
    logger.info("Then, we pass our vocabulary to BERTopic and train the model")

    vectorizer_model= CountVectorizer(vocabulary=vocabulary)
    topic_model = BERTopic(vectorizer_model=vectorizer_model)
    topics, probs = topic_model.fit_transform(docs)

    topic_info = topic_model.get_topic_info()

    for rank, topic_row in enumerate(topic_info.itertuples(), start=1):
        # Safely access columns by index for a DataFrame row tuple
        # topic_info columns are ['Topic', 'Name', 'Count', ...]
        topic_id = int(getattr(topic_row, 'Topic', -1))
        category_name = str(getattr(topic_row, 'Name', f"Topic {topic_id}"))
        representation = getattr(topic_row, 'Representation', [])
        count = float(getattr(topic_row, 'Count', 0))
        doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
        doc_index = doc_indices[0] if doc_indices else -1
        text = docs[doc_index] if doc_index >= 0 else ""
        topic_entry: TopicEntry = {
            "doc_index": doc_index,
            "label": topic_id,
            "count": count,
            "category": category_name,
            "representation": representation,
            "text": text,
        }
        topic_id_suffix = topic_id if topic_id != -1 else "outliers"
        save_file(
            topic_entry,
            f"{OUTPUT_DIR}/topics/topic_{topic_id_suffix}.json"
        )

    save_file(topic_info.to_dict(orient="records"), f"{OUTPUT_DIR}/topic_info.json")