import torch
from wtpsplit import SaT
from typing import List, Optional

def extract_paragraphs(
    text: str, 
    model_name: str = "sat-12l-sm", 
    use_gpu: bool = True, 
    paragraph_threshold: float = 0.5,
    style_or_domain: Optional[str] = None,
    language: str = "en"
) -> List[str]:
    """
    Extracts paragraphs from unstructured text without relying on newline delimiters.
    This function uses the SaT model from wtpsplit to perform semantic segmentation.
    It detects paragraph boundaries based on newline probability predictions,
    making it suitable for noisy or concatenated text (e.g., from PDFs or web scrapes).
    Optimized for Mac M1 MPS, CUDA, or CPU.
    Args:
        text (str): The input text to segment.
        model_name (str, optional): The SaT model to use (e.g., "sat-12l-sm" for high accuracy,
                                    "sat-3l-sm" for faster inference). Defaults to "sat-12l-sm".
        use_gpu (bool, optional): Whether to use GPU (MPS on M1, CUDA elsewhere) if available. Defaults to True.
        paragraph_threshold (float, optional): Threshold for paragraph boundary detection
                                               (higher = more conservative). Defaults to 0.5.
        style_or_domain (Optional[str], optional): LoRA adaptation style/domain (e.g., "ud" for Universal Dependencies).
                                                  Defaults to None (no adaptation).
        language (str, optional): Language for LoRA module (e.g., "en"). Defaults to "en".
    Returns:
        List[str]: A list of extracted paragraphs as strings.
    Raises:
        ValueError: If the model fails to load or text is empty.
    Example:
        >>> text = "This is the first paragraph. It has multiple sentences. This is the second paragraph without newlines."
        >>> extract_paragraphs(text)
        ['This is the first paragraph. It has multiple sentences. ', 'This is the second paragraph without newlines.']
    """
    if not text.strip():
        raise ValueError("Input text cannot be empty.")
    try:
        # Only pass style_or_domain and language if style_or_domain is explicitly provided
        if style_or_domain:
            sat = SaT(model_name, style_or_domain=style_or_domain, language=language)
        else:
            sat = SaT(model_name)
    except Exception as e:
        raise ValueError(f"Failed to load model '{model_name}' with style_or_domain '{style_or_domain}': {e}")
    device = "cpu"
    if use_gpu:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
    if device in ["cuda", "mps"]:
        sat.to(device)
        if device == "cuda":
            sat.half()
    segmented = sat.split(text, do_paragraph_segmentation=True, paragraph_threshold=paragraph_threshold)
    paragraphs = [' '.join(sent.strip() for sent in para) for para in segmented]
    return paragraphs