import torch
from wtpsplit import SaT
from jet.code.markdown_utils import extract_sentences

def run_examples():
    """
    Demonstrates usage of the extract_sentences function with various text inputs.
    Optimized for Mac M1 with MPS support.
    """
    # Example 1: Basic Usage (Single Text Block)
    text = "First sentence without breaks. It continues here. Second sentence starts semantically. More content follows."
    sentences = extract_sentences(text, use_gpu=True)
    print("Example 1 - Basic Usage:")
    print(sentences)
    # Expected: ['First sentence without breaks. It continues here. ', 'Second sentence starts semantically. More content follows.']

    # Example 2: Noisy Text (e.g., PDF OCR Output)
    noisy_text = "ThisisatextwithoutspacesorpunctuationItshouldbedetectedSecondpartbeginsnowWithadifferentsemanticfocus."
    sentences = extract_sentences(noisy_text, model_name="sat-3l-sm", sentence_threshold=0.6, use_gpu=True)
    print("\nExample 2 - Noisy Text:")
    print(sentences)
    # Expected: ~['ThisisatextwithoutspacesorpunctuationItshouldbedetected ', 'SecondpartbeginsnowWithadifferentsemanticfocus.']

    # Example 3: Batch Processing (Multiple Texts)
    sat = SaT("sat-12l-sm")
    if torch.backends.mps.is_available():
        sat.to("mps")  # Leverage M1 MPS
    texts = [
        "Batch text one without newlines. Sentence two.",
        "Another document. Separate sentence."
    ]
    batch_segmented = sat.split(texts, do_paragraph_segmentation=True)
    batch_sentences = [[' '.join(sent.strip() for sent in para) for para in doc] for doc in batch_segmented]
    print("\nExample 3 - Batch Processing:")
    print(batch_sentences)
    # Expected: [['Batch text one without newlines. ', 'Sentence two.'], ['Another document. ', 'Separate sentence.']]

    # Example 4: Domain-Specific Text (e.g., Legal)
    legal_text = "Whereas the parties agree to terms. The agreement shall commence. Notwithstanding prior clauses."
    sentences = extract_sentences(legal_text, model_name="sat-12l", style_or_domain="ud", language="en", use_gpu=True)
    print("\nExample 4 - Legal Text:")
    print(sentences)
    # Expected: ~['Whereas the parties agree to terms. ', 'The agreement shall commence. ', 'Notwithstanding prior clauses.']

if __name__ == "__main__":
    run_examples()