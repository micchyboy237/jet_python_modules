from jet.adapters.bertopic import BERTopic
# from sklearn.datasets import fetch_20newsgroups
from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_data
import os
import shutil

OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

CHUNK_SIZE = 96
CHUNK_OVERLAP = 32

OUTPUT_DIR = f"{OUTPUT_DIR}/chunked_{CHUNK_SIZE}_{CHUNK_OVERLAP}"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

docs = load_sample_data(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
topic_model = BERTopic().fit(docs)

topic_distr, _ = topic_model.approximate_distribution(docs)

save_file(topic_model.visualize_distribution(topic_distr[1]).to_image(), f"{OUTPUT_DIR}/plot_1.png")