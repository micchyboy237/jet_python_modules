import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
from typing import List, Tuple


class SynonymGenerator:
    def __init__(self, model_name: str = "roberta-base"):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(model_name)
        self.model.eval()

    def generate_synonyms(self, sentence: str, target_word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Generate synonyms for a target word in a sentence using RoBERTa.

        Args:
            sentence (str): The input sentence containing the target word.
            target_word (str): The word to find synonyms for.
            top_k (int): Number of synonyms to return.

        Returns:
            List[Tuple[str, float]]: List of (synonym, probability) tuples.
        """
        # Replace target word with <mask>
        if target_word not in sentence:
            return []

        masked_sentence = sentence.replace(
            target_word, self.tokenizer.mask_token)

        # Tokenize input
        inputs = self.tokenizer(masked_sentence, return_tensors="pt")

        # Get mask token index
        mask_token_index = torch.where(
            inputs["input_ids"] == self.tokenizer.mask_token_id)[1]

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get probabilities for masked token
        mask_token_logits = logits[0, mask_token_index, :].squeeze()
        probabilities = torch.softmax(mask_token_logits, dim=-1)

        # Get top k predictions
        top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=-1)

        # Decode tokens to words
        synonyms = []
        for idx, prob in zip(top_k_indices, top_k_probs):
            token = self.tokenizer.decode(
                [idx], skip_special_tokens=True).strip()
            # Ensure the token is a single word and not the original word
            if token and len(token.split()) == 1 and token.lower() != target_word.lower():
                synonyms.append((token, prob.item()))

        return synonyms
