# translators/jp_en.py
from __future__ import annotations

from typing import Literal
from transformers import MarianMTModel, MarianTokenizer
import torch

model_name = "Helsinki-NLP/opus-mt-ja-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

Strategy = Literal["sampling", "diverse_beam", "fast_sampling"]


def translate_ja_en_diverse(
    ja_text: str,
    *,
    n: int = 8,
    strategy: Strategy = "sampling",
    temperature: float = 0.9,
    top_p: float = 0.95,
    diversity_penalty: float = 1.2,
    seed: int | None = None,
) -> list[tuple[str, float]]:
    if seed is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(ja_text, return_tensors="pt", padding=True).to(device)

    # ==================================================================
    # 1. SAMPLING + proper re-scoring (most diverse + real probabilities)
    # ==================================================================
    if strategy == "sampling":
        candidates = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
            num_beams=1,                    # ← THIS FIXES THE ERROR
            max_length=512,
            no_repeat_ngram_size=2,
            early_stopping=False,
        )

        translations = tokenizer.batch_decode(candidates, skip_special_tokens=True)

        # Re-score each candidate for accurate log-probability
        encoded = tokenizer(
            translations,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model(**encoded, labels=encoded.input_ids)
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = encoded.input_ids[..., 1:].contiguous()
            log_probs = torch.gather(
                torch.log_softmax(shift_logits, dim=-1),
                2,
                shift_labels.unsqueeze(-1),
            ).squeeze(-1)

            # Mask padding tokens
            pad_mask = shift_labels != tokenizer.pad_token_id
            log_probs = (log_probs * pad_mask).sum(dim=-1)

        results = list(zip(translations, log_probs.cpu().tolist()))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ==================================================================
    # 2. DIVERSE BEAM SEARCH (deterministic + exact scores)
    # ==================================================================
    elif strategy == "diverse_beam":
        outputs = model.generate(
            **inputs,
            num_beams=n * 2,
            num_beam_groups=n,
            diversity_penalty=diversity_penalty,
            num_return_sequences=n,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        log_probs = transition_scores.sum(1).cpu().tolist()

        translations = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        results = sorted(zip(translations, log_probs), key=lambda x: x[1], reverse=True)
        return results

    # ==================================================================
    # 3. FAST SAMPLING (quick & dirty)
    # ==================================================================
    elif strategy == "fast_sampling":
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
            num_beams=1,                    # ← also fixed here
            max_length=512,
            no_repeat_ngram_size=2,
            early_stopping=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

        approx_scores = (
            outputs.sequences_scores.cpu().tolist()
            if outputs.sequences_scores is not None
            else [0.0] * n
        )
        translations = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        results = sorted(zip(translations, approx_scores), key=lambda x: x[1], reverse=True)
        return results

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


# ——— Test ———
if __name__ == "__main__":
    ja = "今日はとても良い天気ですね。散歩に行きませんか？"

    print("=== sampling (recommended) ===")
    for i, (en, lp) in enumerate(translate_ja_en_diverse(ja, n=8, strategy="sampling"), 1):
        prob = torch.exp(torch.tensor(lp)).item()
        print(f"{i}. {en}")
        print(f"   log_prob={lp:.3f} → prob≈{prob:.4f}\n")