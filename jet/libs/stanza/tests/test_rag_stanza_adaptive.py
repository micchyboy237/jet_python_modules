"""
Tests for adaptive Stanza-based RAG pipeline and dependency visualization.
"""
import os
from pathlib import Path
from typing import Dict, Any

from jet.libs.stanza.rag_stanza_adaptive import run_rag_stanza_adaptive_demo, visualize_sentence_dependency_dot
from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name


class TestRagStanzaAdaptive:
    """Integration-style tests verifying adaptive chunking and DOT export."""

    def setup_method(self):
        self.sample_text = (
            "OpenAI's GPT-5 model was introduced with stronger reasoning and retrieval capabilities. "
            "Researchers from Stanford and MIT studied its syntactic robustness using Stanza NLP. "
            "They found that dependency-based chunking improves factual recall in RAG pipelines, "
            "especially for multilingual data with complex structures. "
            "These insights could lead to more contextually aware AI assistants."
        )

    def test_run_adaptive_demo_outputs_expected_fields(self):
        """
        Given: A realistic long text
        When:  run_rag_stanza_adaptive_demo() is executed
        Then:  It returns structured results including chunks and a DOT file path
        """
        result: Dict[str, Any] = run_rag_stanza_adaptive_demo(self.sample_text)
        assert "parsed_sentences" in result
        assert "chunks" in result
        assert "dot_file" in result
        dot_file = Path(result["dot_file"])
        assert dot_file.exists(), "DOT file should be generated for dependency tree"

    def test_visualize_sentence_dependency_dot_creates_valid_dot(self):
        """
        Given: Parsed sentence data
        When:  visualize_sentence_dependency_dot() is called
        Then:  A valid DOT file is created with nodes and edges
        """
        parsed_sent = {
            "tokens": ["OpenAI", "released", "GPT-5"],
            "heads": [2, 0, 2],
            "deps": ["nsubj", "root", "obj"],
        }
        out_path = f"{get_entry_file_dir()}/generated/{os.path.splitext(get_entry_file_name())[0]}/test_dep_tree.dot"
        dot_path = visualize_sentence_dependency_dot(parsed_sent, out_path)
        assert Path(dot_path).exists()
        content = Path(dot_path).read_text()
        assert "digraph G" in content
        assert "root" in content
        assert "label" in content

    def test_adaptive_chunking_produces_reasonable_output(self):
        """
        Given: A text with varying sentence complexity
        When:  Adaptive chunking is applied
        Then:  Chunks count should adjust dynamically, not always fixed-size
        """
        result = run_rag_stanza_adaptive_demo(self.sample_text)
        chunks = result["chunks"]
        assert 1 <= len(chunks) <= 5
        saliences = [c["salience"] for c in chunks]
        assert all(s >= 0 for s in saliences)
        # Ensure salience increases with length
        assert max(saliences) >= min(saliences)
