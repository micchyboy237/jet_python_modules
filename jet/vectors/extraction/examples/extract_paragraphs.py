from jet.vectors.extraction import extract_paragraphs
from wtpsplit import SaT
import torch

def run_examples():
    """
    Demonstrates usage of the extract_paragraphs function with various text inputs.
    Optimized for Mac M1 with MPS support.
    """
    # Example 1: Basic Usage (Single Text Block)
    text = "First paragraph without breaks. It continues here. Second paragraph starts semantically. More content follows."
    paragraphs = extract_paragraphs(text, use_gpu=True)
    print("Example 1 - Basic Usage:")
    print(paragraphs)
    # Expected: ['First paragraph without breaks. It continues here. ', 'Second paragraph starts semantically. More content follows.']

    # Example 2: Noisy Text (e.g., PDF OCR Output)
    noisy_text = "ThisisatextwithoutspacesorpunctuationItshouldbedetectedSecondpartbeginsnowWithadifferentsemanticfocus."
    paragraphs = extract_paragraphs(noisy_text, model_name="sat-3l-sm", paragraph_threshold=0.6, use_gpu=True)
    print("\nExample 2 - Noisy Text:")
    print(paragraphs)
    # Expected: ~['ThisisatextwithoutspacesorpunctuationItshouldbedetected ', 'SecondpartbeginsnowWithadifferentsemanticfocus.']

    # Example 3: Batch Processing (Multiple Texts)
    sat = SaT("sat-12l-sm")
    if torch.backends.mps.is_available():
        sat.to("mps")  # Leverage M1 MPS
    texts = [
        "Batch text one without newlines. Paragraph two.",
        "Another document. Separate para."
    ]
    batch_segmented = sat.split(texts, do_paragraph_segmentation=True)
    batch_paragraphs = [[' '.join(sent.strip() for sent in para) for para in doc] for doc in batch_segmented]
    print("\nExample 3 - Batch Processing:")
    print(batch_paragraphs)
    # Expected: [['Batch text one without newlines. ', 'Paragraph two.'], ['Another document. ', 'Separate para.']]

    # Example 4: Domain-Specific Text (e.g., Legal)
    legal_text = "Whereas the parties agree to terms. The agreement shall commence. Notwithstanding prior clauses."
    paragraphs = extract_paragraphs(legal_text, model_name="sat-12l", style_or_domain="ud", language="en", use_gpu=True)
    print("\nExample 4 - Legal Text:")
    print(paragraphs)
    # Expected: ~['Whereas the parties agree to terms. ', 'The agreement shall commence. ', 'Notwithstanding prior clauses.']

if __name__ == "__main__":
    run_examples()
