import torch
from tqdm import tqdm
from wtpsplit import SaT
from typing import List, Optional, Union

from jet.wordnet.validators.sentence_validator import is_valid_sentence

# Global cache for a single SaT model instance
_model_cache = {"model": None, "key": None}

def _get_model_key(model_name: str, style_or_domain: Optional[str], language: str) -> tuple:
    """Generate a unique key for the model cache based on configuration."""
    return (model_name, style_or_domain, language)

def _load_model(model_name: str, style_or_domain: Optional[str], language: str) -> SaT:
    """Load or retrieve cached SaT model, ensuring only one model is stored."""
    model_key = _get_model_key(model_name, style_or_domain, language)
    if _model_cache["key"] != model_key:
        try:
            if style_or_domain:
                sat = SaT(model_name, style_or_domain=style_or_domain, language=language)
            else:
                sat = SaT(model_name)
            _model_cache["model"] = sat
            _model_cache["key"] = model_key
        except Exception as e:
            raise ValueError(f"Failed to load model '{model_name}' with style_or_domain '{style_or_domain}': {e}")
    return _model_cache["model"]

def extract_sentences(
    text: Union[str, List[str]], 
    model_name: str = "sat-12l-sm", 
    use_gpu: bool = True, 
    sentence_threshold: float = 0.5,
    style_or_domain: Optional[str] = None,
    language: str = "en",
    valid_only: bool = False
) -> List[str]:
    """
    Extracts sentences from unstructured text without relying on newline delimiters.
    This function uses the SaT model from wtpsplit to perform semantic segmentation.
    It detects sentence boundaries based on newline probability predictions,
    making it suitable for noisy or concatenated text (e.g., from PDFs or web scrapes).
    Optimized for Mac M1 MPS, CUDA, or CPU.
    Args:
        text (str): The input text to segment.
        model_name (str, optional): The SaT model to use (e.g., "sat-12l-sm" for high accuracy,
                                    "sat-3l-sm" for faster inference). Defaults to "sat-12l-sm".
        use_gpu (bool, optional): Whether to use GPU (MPS on M1, CUDA elsewhere) if available. Defaults to True.
        sentence_threshold (float, optional): Threshold for sentence boundary detection
                                             (higher = more conservative). Defaults to 0.5.
        style_or_domain (Optional[str], optional): LoRA adaptation style/domain (e.g., "ud" for Universal Dependencies).
                                                  Defaults to None (no adaptation).
        language (str, optional): Language for LoRA module (e.g., "en"). Defaults to "en".
        valid_only (bool, optional): If True, only keep sentences passing a validator
                                      (filters out e.g. fragments or very short/incomplete sentences).
                                      Defaults to False.
    Returns:
        List[str]: A list of extracted sentences as strings.
    Raises:
        ValueError: If the model fails to load or text is empty.
    Example:
        >>> text = "This is the first sentence. It has multiple parts. This is the second sentence without newlines."
        >>> extract_sentences(text)
        ['This is the first sentence. It has multiple parts. ', 'This is the second sentence without newlines.']
    """
    if not text.strip() if isinstance(text, str) else not "".join(text).strip():
        return []
    
    sat = _load_model(model_name, style_or_domain, language)
    
    device = "cpu"
    if use_gpu:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
    
    # Move model to device only if it hasn't been moved already
    if sat.device != device:
        sat.to(device)
        if device == "cuda":
            sat.half()
    
    # Normalize input: if list of strings, join into one string for SaT
    if isinstance(text, list):
        input_text = "\n\n".join(text)  # Preserve paragraph structure
    else:
        input_text = text

    segmented = sat.split(input_text, do_paragraph_segmentation=True, paragraph_threshold=sentence_threshold)

    # Robust sentence joining: handle cases where para is str or list of str
    def join_para(para):
        if isinstance(para, str):
            return para.strip()
        else:
            return ' '.join(s.strip() for s in para if isinstance(s, str))

    sentences = [join_para(para) for para in segmented if join_para(para)]
    if valid_only:
        # Filter valid sentences
        sentences = [s for s in tqdm(sentences, desc="Filtering valid sentences") if is_valid_sentence(s)]
    return sentences