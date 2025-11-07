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
    # Example 1: Basic Usage
    text = (
        "First paragraph begins here. It contains multiple sentences without newline breaks. "
        "The model must detect semantic boundaries. This is the final sentence of the first block.\n\n"
        
        "Second paragraph starts after a clear separation. Here we test if the model respects "
        "existing newlines as strong paragraph cues. Another sentence follows in the same paragraph.\n\n"
        
        "Third and final paragraph. Short but valid. Contains only two sentences."
    )
    sentences = extract_sentences(text, use_gpu=True)
    print("Example 1 - Basic Usage:")
    save_file(sentences, f"{OUTPUT_DIR}/1_basic_usage_results.json")

    # Example 2.1: Paragraph Segmentation (Low Threshold)
    paragraphs = extract_sentences(text, do_paragraph_segmentation=True, paragraph_threshold=0.1, use_gpu=True)
    print("Example 2.1 - Paragraph Segmentation (Low Threshold):")
    save_file(paragraphs, f"{OUTPUT_DIR}/2_1_paragraph_low_threshold_results.json")

    # Example 2.2: Paragraph Segmentation (Medium Threshold)
    paragraphs = extract_sentences(text, do_paragraph_segmentation=True, paragraph_threshold=0.5, use_gpu=True)
    print("Example 2.2 - Paragraph Segmentation (Medium Threshold):")
    save_file(paragraphs, f"{OUTPUT_DIR}/2_2_paragraph_medium_threshold_results.json")

    # Example 2.3: Paragraph Segmentation (High Threshold)
    paragraphs = extract_sentences(text, do_paragraph_segmentation=True, paragraph_threshold=0.9, use_gpu=True)
    print("Example 2.3 - Paragraph Segmentation (High Threshold):")
    save_file(paragraphs, f"{OUTPUT_DIR}/2_3_paragraph_high_threshold_results.json")

    # Example 3: Noisy Text (e.g., PDF OCR Output)
    noisy_text = "ThisisatextwithoutspacesorpunctuationItshouldbedetectedSecondpartbeginsnowWithadifferentsemanticfocus."
    sentences = extract_sentences(noisy_text, do_paragraph_segmentation=True, paragraph_threshold=0.6, use_gpu=True)
    print("\nExample 3 - Noisy Text:")
    save_file(sentences, f"{OUTPUT_DIR}/3_noisy_text_results.json")
    # Expected: ~['ThisisatextwithoutspacesorpunctuationItshouldbedetected ', 'SecondpartbeginsnowWithadifferentsemanticfocus.']

    # Example 4.1: Batch Processing (Basic Usage)
    texts = [
        "Batch text one without newlines. Sentence two.",
        "Another document. Separate sentence."
    ]
    batch_segmented = extract_sentences(texts, use_gpu=True)
    print("\nExample 4.1 - Batch Processing:")
    save_file(batch_segmented, f"{OUTPUT_DIR}/4_1_batch_processing_results.json")

    # Example 4.2: Batch Processing (Paragraph Segmentation)
    batch_segmented = extract_sentences(texts, do_paragraph_segmentation=True, use_gpu=True)
    print("\nExample 4.2 - Batch Processing (Paragraph Segmentation):")
    save_file(batch_segmented, f"{OUTPUT_DIR}/4_2_batch_processing_paragraph_results.json")

    # Example 5: Domain-Specific Text (e.g., Legal)
    legal_text = "Whereas the parties agree to terms. The agreement shall commence. Notwithstanding prior clauses."
    sentences = extract_sentences(legal_text, style_or_domain="ud", language="en", use_gpu=True)
    print("\nExample 5 - Legal Text:")
    save_file(sentences, f"{OUTPUT_DIR}/5_legal_text_results.json")
    # Expected: ~['Whereas the parties agree to terms. ', 'The agreement shall commence. ', 'Notwithstanding prior clauses.']

if __name__ == "__main__":
    run_examples()