from jet.adapters.keybert import KeyBERT
from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_data
import os
import shutil

OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

CHUNK_SIZE = 128
CHUNK_OVERLAP = 32

OUTPUT_DIR = f"{OUTPUT_DIR}/chunked_{CHUNK_SIZE}_{CHUNK_OVERLAP}"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Prepare documents 
# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
docs = load_sample_data(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

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

labels = topic_model.generate_topic_labels(nr_words=1)
save_file(labels, f"{OUTPUT_DIR}/labels.json")