from typing import List, Union
from jet.code.extraction import extract_sentences
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def extract_rag_contexts(text: Union[str, List[str]]) -> List[str]:
    """
    Demonstrates usage of the extract_sentences function with various text inputs.
    Optimized for Mac M1 with MPS support.
    """
    if isinstance(text, str):
        text = [text]

    contexts = []
    for t in text:
        sentences = extract_sentences(t, use_gpu=True)

if __name__ == "__main__":
    extract_rag_contexts()
