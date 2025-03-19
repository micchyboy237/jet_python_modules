from fastapi import APIRouter, HTTPException
from jet.file.utils import load_file
from jet.search.similarity import get_bm25_similarities
from jet.search.transformers import clean_string
from pydantic import BaseModel
from typing import List
from jet.wordnet.words import get_words
from shared.data_types.job import JobData
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

phrase_detector = None
colbert_model = None
bert_model = None
cohere_model = None
t5_model = None
t5_tokenizer = None


def setup_phrase_detector(sentences):
    from jet.wordnet.gensim_scripts.phrase_detector import PhraseDetector

    global phrase_detector

    if not phrase_detector:
        phrase_detector = PhraseDetector(
            PHRASE_MODEL_PATH, sentences, reset_cache=True)

    return phrase_detector


# Load job data
def load_data(file_path: str) -> List[JobData]:
    return load_file(file_path)


# Extract job descriptions for ranking
def prepare_sentences(data: List[JobData]) -> List[str]:
    return [
        "\n".join([
            job["title"],
            job["details"],
            " ".join(
                sorted(job["entities"]["technology_stack"], key=str.lower)),
            " ".join(sorted(job["tags"], key=str.lower)),
        ]).lower()
        for job in data
    ]


# Load Sentence Transformers Models
def setup_colbert_model():
    global colbert_model

    if not colbert_model:
        colbert_model = SentenceTransformer("colbert-ir/colbertv2.0")

    return colbert_model


def setup_bert_model():
    global bert_model

    if not bert_model:
        bert_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2')

    return bert_model


def setup_cohere_model():
    import cohere

    global cohere_model

    if not cohere_model:
        # Initialize Cohere API (Replace with your API key)
        cohere_model = cohere.Client("your-cohere-api-key")

    return cohere_model


def setup_t5_model():
    global t5_model, t5_tokenizer

    model = "castorini/monot5-small-msmarco-10k"
    # model = "ramsrigouthamg/t5-small-relevance"
    # model = "boost/pt-t5-small-msmarco"

    if t5_model is None:
        t5_tokenizer = AutoTokenizer.from_pretrained(model)
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(model)

    return t5_model, t5_tokenizer
