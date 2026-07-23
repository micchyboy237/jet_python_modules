"""Generic hierarchical reduction.

Repeatedly groups a list of texts into token-budget-safe batches and reduces
each batch to one text, until a single text remains. This is the one piece of
logic shared by both file-chunk summarization (many chunks -> one file
summary) and folder/tree merging (many child summaries -> one folder
summary) -- it is the "repeat until it fits" step of map-reduce.
"""

import logging
from typing import Callable, List

logger = logging.getLogger("md_summarizer.reduce")


def _pack_into_batches(items: List[str], token_counter: Callable[[str], int], budget: int) -> List[List[str]]:
    """Greedily group items into batches whose combined token count stays
    under budget. An item larger than the whole budget becomes its own
    single-item batch (callers are expected to have already chunked any
    individually oversized input before it reaches this function)."""
    batches: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0
    for item in items:
        item_tokens = token_counter(item)
        if current and current_tokens + item_tokens > budget:
            batches.append(current)
            current = [item]
            current_tokens = item_tokens
        else:
            current.append(item)
            current_tokens += item_tokens
    if current:
        batches.append(current)
    return batches


def hierarchical_reduce(
    items: List[str],
    reduce_call: Callable[[List[str]], str],
    token_counter: Callable[[str], int],
    token_budget: int,
    node_label: str = "root",
    max_levels: int = 6,
) -> str:
    """Collapse `items` down to a single string by repeatedly batching + reducing.

    Level 0: pack raw items into budget-safe batches, reduce each batch to get
    a new (shorter) list of items. Repeat on that list until only one item
    remains, or everything now fits in a single reduce call. `max_levels` is a
    safety valve against runaway recursion on pathological inputs (e.g. a
    reduce call that doesn't actually shrink its input).
    """
    if not items:
        raise ValueError(f"[{node_label}] hierarchical_reduce called with no items")

    current_items = items
    level = 0
    while True:
        total_tokens = sum(token_counter(i) for i in current_items)

        if len(current_items) == 1:
            logger.info("[%s] reduce complete at level %d (single item remains)", node_label, level)
            return current_items[0]

        if total_tokens <= token_budget:
            logger.info(
                "[%s] level %d: %d items fit in one call (%d tokens <= budget %d) -> final reduce",
                node_label, level, len(current_items), total_tokens, token_budget,
            )
            return reduce_call(current_items)

        if level >= max_levels:
            logger.warning(
                "[%s] hit max_levels=%d with %d items still unreduced; forcing one last combined reduce",
                node_label, max_levels, len(current_items),
            )
            return reduce_call(current_items)

        batches = _pack_into_batches(current_items, token_counter, token_budget)
        logger.info(
            "[%s] level %d: %d items (%d tokens) exceed budget %d -> packed into %d batch(es)",
            node_label, level, len(current_items), total_tokens, token_budget, len(batches),
        )
        current_items = [reduce_call(batch) for batch in batches]
        level += 1
