"""Recursive directory walk + bottom-up orchestration.

Discovers all .md files under a target directory, builds a tree mirroring the
folder structure, and processes it bottom-up: files are mapped to summaries,
then each folder is reduced from its children (files + subfolders), then the
root's children are synthesized into the final digest.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .agents import reduce_folder, summarize_file, synthesize_root, verify_digest
from .config import PipelineConfig
from .llm_client import LLMClient

logger = logging.getLogger("md_summarizer.pipeline")


@dataclass
class FolderNode:
    path: Path
    subfolders: List["FolderNode"] = field(default_factory=list)
    md_files: List[Path] = field(default_factory=list)

    # populated during processing, kept around for the final report + verifier
    file_summaries: Dict[str, str] = field(default_factory=dict)
    folder_summary: Optional[str] = None


def discover_tree(root_dir: Path) -> FolderNode:
    """Build a FolderNode tree of every directory under root_dir that
    contains, directly or transitively, at least one .md file. Directories
    with no markdown anywhere beneath them are pruned rather than kept as
    dead branches."""

    def build(dir_path: Path) -> Optional[FolderNode]:
        node = FolderNode(path=dir_path)
        try:
            entries = sorted(dir_path.iterdir())
        except PermissionError as exc:
            logger.warning("skipping %s: %s", dir_path, exc)
            return None

        for entry in entries:
            if entry.is_dir():
                child = build(entry)
                if child is not None:
                    node.subfolders.append(child)
            elif entry.suffix.lower() == ".md":
                node.md_files.append(entry)

        if not node.subfolders and not node.md_files:
            return None
        return node

    root = build(root_dir)
    if root is None:
        raise ValueError(f"No markdown files found anywhere under {root_dir}")
    return root


def process_tree(
    node: FolderNode,
    llm: LLMClient,
    config: PipelineConfig,
    is_root: bool = False,
) -> str:
    """Bottom-up: map every file in this folder, recurse into subfolders
    first, then reduce (or synthesize, at the root) this folder's children
    into one summary."""
    budget = config.input_token_budget
    max_out = config.reserved_output_tokens

    for md_file in node.md_files:
        label = str(md_file)
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.error("could not read %s: %s -- skipping", md_file, exc)
            continue
        logger.info("mapping file: %s", label)
        summary = summarize_file(llm, label, text, budget, max_out, config.temperature)
        node.file_summaries[label] = summary

    child_summaries: List[str] = list(node.file_summaries.values())
    for sub in node.subfolders:
        child_summaries.append(process_tree(sub, llm, config, is_root=False))

    child_summaries = [s for s in child_summaries if s]
    if not child_summaries:
        logger.warning("folder %s has no summarizable content", node.path)
        node.folder_summary = ""
        return ""

    label = str(node.path)
    if is_root:
        logger.info("synthesizing final digest at root: %s", label)
        result = synthesize_root(llm, child_summaries, budget, max_out, config.temperature)
    else:
        logger.info("reducing folder: %s (%d children)", label, len(child_summaries))
        result = reduce_folder(llm, label, child_summaries, budget, max_out, config.temperature)

    node.folder_summary = result
    return result


def collect_all_file_summaries(node: FolderNode) -> List[str]:
    """Flatten every leaf file summary in the tree -- used for the verifier's
    sampling step."""
    summaries = [s for s in node.file_summaries.values() if s]
    for sub in node.subfolders:
        summaries.extend(collect_all_file_summaries(sub))
    return summaries


def run_pipeline(
    root_dir: Path,
    llm: LLMClient,
    config: PipelineConfig,
    run_verification: bool = True,
):
    """End-to-end: discover -> process bottom-up -> (optional) verify.
    Returns (final_digest, verification_report_or_None, tree)."""
    logger.info("discovering markdown files under %s", root_dir)
    tree = discover_tree(root_dir)

    logger.info("starting bottom-up processing")
    final_digest = process_tree(tree, llm, config, is_root=True)

    verification_report = None
    if run_verification:
        leaf_summaries = collect_all_file_summaries(tree)
        if leaf_summaries:
            logger.info("running verifier against %d leaf summaries", len(leaf_summaries))
            verification_report = verify_digest(
                llm,
                final_digest,
                leaf_summaries,
                config.input_token_budget,
                config.reserved_output_tokens,
                config.temperature,
                sample_size=config.verify_sample_size,
            )

    return final_digest, verification_report, tree
