from typing import List, Dict, Optional, TypedDict
from jet.llm.mlx.config import DEFAULT_MODEL
from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.models import resolve_model
from jet.logger import logger
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.utils import TokenizerWrapper


class LabelResult(TypedDict):
    label: str
    token_id: int
    success: bool
    error: Optional[str]


def label_multiple_choice(
    instruction: str,
    options: List[str],
    model_path: ModelType = DEFAULT_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 1
) -> LabelResult:
    model_path = resolve_model(model_path)
    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        return LabelResult(label="", token_id=-1, success=False, error=f"Model load failed: {e}")

    messages = [
        {"role": "system", "content": f"You are a labelling assistant. Choose one label from the options provided: {', '.join(options)}."
         },
        {"role": "user", "content": instruction}
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        return LabelResult(label="", token_id=-1, success=False, error=f"Prompt formatting failed: {e}")

    try:
        option_tokens = []
        for option in options:
            token = tokenizer.encode(option, add_special_tokens=False)[0]
            option_tokens.append(token)
            logger.debug(f"Label token for '{option}': {token}")

        logit_bias: Dict[int, float] = {token: 0.0 for token in option_tokens}
        logits_processors = make_logits_processors(logit_bias=logit_bias)
        sampler = make_sampler(temp=temperature, top_p=top_p)
        stop_tokens = tokenizer.encode("\n") + list(tokenizer.eos_token_ids)

        for output in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            sampler=sampler
        ):
            if output.token in stop_tokens:
                return LabelResult(label="", token_id=output.token, success=False, error="Received stop token.")
            label = output.text
            token_id = output.token
            if label not in options:
                return LabelResult(label=label, token_id=token_id, success=False, error="Invalid label generated.")
            return LabelResult(label=label, token_id=token_id, success=True, error=None)

        return LabelResult(label="", token_id=-1, success=False, error="No output generated.")

    except Exception as e:
        return LabelResult(label="", token_id=-1, success=False, error=str(e))
