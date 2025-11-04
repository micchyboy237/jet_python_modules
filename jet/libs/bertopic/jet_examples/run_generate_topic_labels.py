from jet.adapters.keybert import KeyBERT
from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_data
import os
import shutil

OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Prepare documents 
# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
docs = load_sample_data()

# Extract keywords
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(docs)
save_file(keywords, f"{OUTPUT_DIR}/keywords.json")

# Create our vocabulary
vocabulary = [k[0] for keyword in keywords for k in keyword]
vocabulary = list(set(vocabulary))
save_file(vocabulary, f"{OUTPUT_DIR}/vocabulary.json")
# Then, we pass our vocabulary to BERTopic and train the model:

from jet.adapters.bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model= CountVectorizer(vocabulary=vocabulary)
topic_model = BERTopic(vectorizer_model=vectorizer_model)
topics, probs = topic_model.fit_transform(docs)

labels = topic_model.generate_topic_labels(nr_words=3)
save_file(labels, f"{OUTPUT_DIR}/labels.json")