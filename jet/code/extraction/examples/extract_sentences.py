import os
import shutil
from jet.code.extraction import extract_sentences
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def run_examples():
    """
    Demonstrates usage of the extract_sentences function with various text inputs.
    Optimized for Mac M1 with MPS support.
    """
    # Example 1: Basic Usage (Single Text Block)
    text = "First sentence without breaks. It continues here. Second sentence starts semantically. More content follows."
    sentences = extract_sentences(text, use_gpu=True)
    print("Example 1 - Basic Usage:")
    save_file(sentences, f"{OUTPUT_DIR}/1_basic_usage_results.json")
    # Expected: ['First sentence without breaks. It continues here. ', 'Second sentence starts semantically. More content follows.']

    # Example 2: Noisy Text (e.g., PDF OCR Output)
    noisy_text = "ThisisatextwithoutspacesorpunctuationItshouldbedetectedSecondpartbeginsnowWithadifferentsemanticfocus."
    sentences = extract_sentences(noisy_text, do_paragraph_segmentation=True, paragraph_threshold=0.6, use_gpu=True)
    print("\nExample 2 - Noisy Text:")
    save_file(sentences, f"{OUTPUT_DIR}/2_noisy_text_results.json")
    # Expected: ~['ThisisatextwithoutspacesorpunctuationItshouldbedetected ', 'SecondpartbeginsnowWithadifferentsemanticfocus.']

    # Example 3: Batch Processing (Multiple Texts)
    texts = [
        "Batch text one without newlines. Sentence two.",
        "Another document. Separate sentence."
    ]
    batch_segmented = extract_sentences(texts, do_paragraph_segmentation=True, use_gpu=True)
    batch_sentences = [[' '.join(sent.strip() for sent in para) for para in doc] for doc in batch_segmented]
    print("\nExample 3 - Batch Processing:")
    save_file(batch_sentences, f"{OUTPUT_DIR}/3_batch_processing_results.json")
    # Expected: [['Batch text one without newlines. ', 'Sentence two.'], ['Another document. ', 'Separate sentence.']]

    # Example 4: Domain-Specific Text (e.g., Legal)
    legal_text = "Whereas the parties agree to terms. The agreement shall commence. Notwithstanding prior clauses."
    sentences = extract_sentences(legal_text, style_or_domain="ud", language="en", use_gpu=True)
    print("\nExample 4 - Legal Text:")
    save_file(sentences, f"{OUTPUT_DIR}/4_legal_text_results.json")
    # Expected: ~['Whereas the parties agree to terms. ', 'The agreement shall commence. ', 'Notwithstanding prior clauses.']

if __name__ == "__main__":
    run_examples()