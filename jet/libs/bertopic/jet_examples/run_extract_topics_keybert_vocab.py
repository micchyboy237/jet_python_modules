
from typing import List, TypedDict
from jet.file.utils import save_file
from jet.logger import logger
import numpy as np
import os
import shutil


OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

class TopicCategory(TypedDict):
    label: int
    category: str
    representation: List[str]
    count: int

class TopicDocProb(TypedDict):
    doc_index: int
    prob: float
    text: str

class TopicEntry(TopicCategory):
    doc_probs: List[TopicDocProb]

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

    # Convert probs to List[float]
    probs_list: List[float] = probs.tolist() if isinstance(probs, np.ndarray) else list(probs)

    topic_info = topic_model.get_topic_info()

    all_topics = []
    for topic_row in topic_info.itertuples():
        topic_id = int(getattr(topic_row, 'Topic', -1))
        category_name = str(getattr(topic_row, 'Name', f"Topic {topic_id}"))
        representation = getattr(topic_row, 'Representation', [])
        count = int(getattr(topic_row, 'Count', 0))

        # Build doc_probs: List[TopicDocProb]
        doc_prob_list: List[TopicDocProb] = []
        for i, assigned_topic in enumerate(topics):
            if assigned_topic == topic_id:
                prob = probs_list[i]
                if topic_id == -1:
                    prob = 0.0
                doc_prob_list.append({
                    "doc_index": i,
                    "prob": float(prob),
                    "text": docs[i]
                })
        doc_prob_list.sort(key=lambda x: x["prob"], reverse=True)
        doc_prob_list = doc_prob_list[:10]

        topic_entry: TopicEntry = {
            "label": topic_id,
            "category": category_name,
            "representation": representation,
            "count": count,
            "doc_probs": doc_prob_list
        }

        topic_id_suffix = topic_id if topic_id != -1 else "outliers"
        all_topics.append(topic_entry)
        save_file(
            topic_entry,
            f"{OUTPUT_DIR}/topics/topic_{topic_id_suffix}.json"
        )

    save_file(all_topics, f"{OUTPUT_DIR}/topics.json")
    save_file(topic_info.to_dict(orient="records"), f"{OUTPUT_DIR}/topic_info.json")
