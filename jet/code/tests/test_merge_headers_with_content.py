import pytest
from typing import List, Dict, Any
from jet.code.markdown_utils import merge_headers_with_content

# Type alias for MarkdownToken
MarkdownToken = Dict[str, Any]


class TestMergeHeadersWithContent:
    def test_merge_single_header_with_paragraph(self):
        # Given: A list of tokens with one header followed by a paragraph
        input_tokens: List[MarkdownToken] = [
            {'type': 'header', 'content': 'Header 1',
                'level': 1, 'meta': {}, 'line': 1},
            {'type': 'paragraph', 'content': 'Paragraph content', 'line': 2}
        ]
        expected: List[MarkdownToken] = [
            {
                'type': 'header',
                'content': 'Header 1\nParagraph content',
                'level': 1,
                'meta': {},
                'line': 1
            }
        ]

        # When: Merging headers with content
        result = merge_headers_with_content(input_tokens)

        # Then: The header and paragraph are merged into a single header token
        assert result == expected, "Expected header and paragraph to be merged"

    def test_merge_header_with_unordered_list(self):
        # Given: A header followed by an unordered list
        input_tokens: List[MarkdownToken] = [
            {'type': 'header', 'content': 'Header 1',
                'level': 1, 'meta': None, 'line': 1},
            {
                'type': 'unordered_list',
                'content': None,
                'meta': {'items': [{'text': 'Item 1'}, {'text': 'Item 2'}]},
                'line': 2
            }
        ]
        expected: List[MarkdownToken] = [
            {
                'type': 'header',
                'content': 'Header 1\n- Item 1\n- Item 2',
                'level': 1,
                'meta': None,
                'line': 1
            }
        ]

        # When: Merging headers with content
        result = merge_headers_with_content(input_tokens)

        # Then: The header and unordered list are merged correctly
        assert result == expected, "Expected header and unordered list to be merged"

    def test_merge_header_with_ordered_list(self):
        # Given: A header followed by an ordered list
        input_tokens: List[MarkdownToken] = [
            {'type': 'header', 'content': 'Header 1',
                'level': 1, 'meta': {}, 'line': 1},
            {
                'type': 'ordered_list',
                'content': None,
                'meta': {'items': [{'text': 'Item 1'}, {'text': 'Item 2'}]},
                'line': 2
            }
        ]
        expected: List[MarkdownToken] = [
            {
                'type': 'header',
                'content': 'Header 1\n1. Item 1\n2. Item 2',
                'level': 1,
                'meta': {},
                'line': 1
            }
        ]

        # When: Merging headers with content
        result = merge_headers_with_content(input_tokens)

        # Then: The header and ordered list are merged correctly
        assert result == expected, "Expected header and ordered list to be merged"

    def test_multiple_headers_with_mixed_content(self):
        # Given: Multiple headers with paragraphs, unordered lists, and ordered lists
        input_tokens: List[MarkdownToken] = [
            {'type': 'header', 'content': 'Header 1',
                'level': 1, 'meta': {}, 'line': 1},
            {'type': 'paragraph', 'content': 'Paragraph 1', 'line': 2},
            {
                'type': 'unordered_list',
                'content': None,
                'meta': {'items': [{'text': 'List item'}]},
                'line': 3
            },
            {'type': 'header', 'content': 'Header 2',
                'level': 2, 'meta': {}, 'line': 4},
            {'type': 'paragraph', 'content': 'Paragraph 2', 'line': 5},
            {
                'type': 'ordered_list',
                'content': None,
                'meta': {'items': [{'text': 'Ordered item'}]},
                'line': 6
            }
        ]
        expected: List[MarkdownToken] = [
            {
                'type': 'header',
                'content': 'Header 1\nParagraph 1\n- List item',
                'level': 1,
                'meta': {},
                'line': 1
            },
            {
                'type': 'header',
                'content': 'Header 2\nParagraph 2\n1. Ordered item',
                'level': 2,
                'meta': {},
                'line': 4
            }
        ]

        # When: Merging headers with content
        result = merge_headers_with_content(input_tokens)

        # Then: All headers are merged with their respective content
        assert result == expected, "Expected multiple headers to be merged with mixed content"

    def test_non_header_before_first_header(self):
        # Given: A paragraph before the first header
        input_tokens: List[MarkdownToken] = [
            {'type': 'paragraph', 'content': 'Orphan paragraph', 'line': 1},
            {'type': 'header', 'content': 'Header 1',
                'level': 1, 'meta': {}, 'line': 2},
            {'type': 'paragraph', 'content': 'Paragraph content', 'line': 3}
        ]
        expected: List[MarkdownToken] = [
            {
                'type': 'header',
                'content': 'Header 1\nParagraph content',
                'level': 1,
                'meta': {},
                'line': 2
            }
        ]

        # When: Merging headers with content
        result = merge_headers_with_content(input_tokens)

        # Then: Non-header tokens before the first header are ignored
        assert result == expected, "Expected non-header tokens before first header to be ignored"

    def test_header_with_no_content(self):
        # Given: A single header with no following content
        input_tokens: List[MarkdownToken] = [
            {'type': 'header', 'content': 'Header 1',
                'level': 1, 'meta': {}, 'line': 1}
        ]
        expected: List[MarkdownToken] = [
            {
                'type': 'header',
                'content': 'Header 1',
                'level': 1,
                'meta': {},
                'line': 1
            }
        ]

        # When: Merging headers with content
        result = merge_headers_with_content(input_tokens)

        # Then: The header is preserved as is
        assert result == expected, "Expected single header to be preserved"

    def test_empty_token_list(self):
        # Given: An empty token list
        input_tokens: List[MarkdownToken] = []
        expected: List[MarkdownToken] = []

        # When: Merging headers with content
        result = merge_headers_with_content(input_tokens)

        # Then: An empty list is returned
        assert result == expected, "Expected empty list for empty input"

    def test_header_with_code_block(self):
        # Given: A header followed by a code block
        input_tokens: List[MarkdownToken] = [
            {'type': 'header', 'content': 'Header 1',
                'level': 1, 'meta': {}, 'line': 1},
            {'type': 'code_block', 'content': 'print("Hello")', 'meta': {
                'language': 'python'}, 'line': 2}
        ]
        expected: List[MarkdownToken] = [
            {
                'type': 'header',
                'content': 'Header 1\nprint("Hello")',
                'level': 1,
                'meta': {},
                'line': 1
            }
        ]

        # When: Merging headers with content
        result = merge_headers_with_content(input_tokens)

        # Then: The header and code block are merged correctly
        assert result == expected, "Expected header and code block to be merged"

    def test_provided_input(self):
        # Given: The provided input with header, paragraphs, and ordered list
        input_tokens: List[MarkdownToken] = [
            {
                'content': '10 RAG Papers You Should Read from February 2025',
                'level': 1,
                'line': 24,
                'type': 'header'
            },
            {
                'content': 'Research',
                'line': 26,
                'type': 'paragraph'
            },
            {
                'content': "We have compiled a list of 10 research papers on RAG published in February. If you're interested in learning about the developments happening in RAG, you'll find these papers insightful.",
                'line': 28,
                'type': 'paragraph'
            },
            {
                'content': 'Out of all the papers on RAG published in February, these ones caught our eye:',
                'line': 30,
                'type': 'paragraph'
            },
            {
                'line': 32,
                'meta': {
                    'items': [
                        {'text': 'DeepRAG: Introduces a Markov Decision Process (MDP) approach to retrieval, allowing adaptive knowledge retrieval that improves answer accuracy by 21.99%.', 'task_item': False},
                        {'text': 'SafeRAG: A benchmark assessing security vulnerabilities in RAG systems, identifying critical weaknesses across 14 different RAG components.', 'task_item': False},
                        {'text': 'RAG vs. GraphRAG: A systematic comparison of text-based RAG and GraphRAG, highlighting how structured knowledge graphs can enhance retrieval performance.', 'task_item': False},
                        {'text': 'Towards Fair RAG: Investigates fair ranking techniques in RAG retrieval, demonstrating how fairness-aware retrieval can improve source attribution without compromising performance.', 'task_item': False},
                        {'text': 'From RAG to Memory: Introduces HippoRAG 2, which enhances retrieval and improves long-term knowledge retention, making AI reasoning more human-like.', 'task_item': False},
                        {'text': 'MEMERAG: A multilingual evaluation benchmark for RAG, ensuring faithfulness and relevance across multiple languages with expert annotations.', 'task_item': False},
                        {'text': 'Judge as a Judge: Proposes ConsJudge, a method that improves LLM-based evaluation of RAG models using consistency-driven training.', 'task_item': False},
                        {'text': 'Does RAG Really Perform Bad in Long-Context Processing?: Introduces RetroLM, a retrieval method that optimizes long-context comprehension while reducing computational costs.', 'task_item': False},
                        {'text': 'RankCoT RAG: A Chain-of-Thought (CoT) based approach to refine RAG knowledge retrieval, filtering out irrelevant documents for more precise AI-generated responses.', 'task_item': False},
                        {'text': 'Mitigating Bias in RAG: Analyzes how biases from LLMs, embedders, proposes reverse-biasing the embedder to reduce unwanted bias.', 'task_item': False}
                    ]
                },
                'type': 'ordered_list'
            }
        ]
        expected: List[MarkdownToken] = [
            {
                'type': 'header',
                'content': '10 RAG Papers You Should Read from February 2025\nResearch\nWe have compiled a list of 10 research papers on RAG published in February. If you\'re interested in learning about the developments happening in RAG, you\'ll find these papers insightful.\nOut of all the papers on RAG published in February, these ones caught our eye:\n1. DeepRAG: Introduces a Markov Decision Process (MDP) approach to retrieval, allowing adaptive knowledge retrieval that improves answer accuracy by 21.99%.\n2. SafeRAG: A benchmark assessing security vulnerabilities in RAG systems, identifying critical weaknesses across 14 different RAG components.\n3. RAG vs. GraphRAG: A systematic comparison of text-based RAG and GraphRAG, highlighting how structured knowledge graphs can enhance retrieval performance.\n4. Towards Fair RAG: Investigates fair ranking techniques in RAG retrieval, demonstrating how fairness-aware retrieval can improve source attribution without compromising performance.\n5. From RAG to Memory: Introduces HippoRAG 2, which enhances retrieval and improves long-term knowledge retention, making AI reasoning more human-like.\n6. MEMERAG: A multilingual evaluation benchmark for RAG, ensuring faithfulness and relevance across multiple languages with expert annotations.\n7. Judge as a Judge: Proposes ConsJudge, a method that improves LLM-based evaluation of RAG models using consistency-driven training.\n8. Does RAG Really Perform Bad in Long-Context Processing?: Introduces RetroLM, a retrieval method that optimizes long-context comprehension while reducing computational costs.\n9. RankCoT RAG: A Chain-of-Thought (CoT) based approach to refine RAG knowledge retrieval, filtering out irrelevant documents for more precise AI-generated responses.\n10. Mitigating Bias in RAG: Analyzes how biases from LLMs, embedders, proposes reverse-biasing the embedder to reduce unwanted bias.',
                'level': 1,
                'meta': {},
                'line': 24
            }
        ]

        # When: Merging headers with content
        result = merge_headers_with_content(input_tokens)

        # Then: The header, paragraphs, and ordered list are merged correctly
        assert result == expected, "Expected header, paragraphs, and ordered list to be merged"
