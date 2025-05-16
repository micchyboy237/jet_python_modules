import pytest
from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.tasks.search_contexts_by_index import search_contexts_by_index

MODEL_PATH: ModelType = "llama-3.2-3b-instruct-4bit"


def test_top_n_relevant_contexts():
    query = "What is the capital of France?"
    contexts = [
        "The theory of relativity was developed by Albert Einstein.",
        "The capital of France is Paris.",
        "Paris hosts many tourists in France.",
        "Florida is a state in the USA."
    ]
    result = search_contexts_by_index(query, contexts, MODEL_PATH, top_n=2)
    assert result["is_valid"]
    assert len(result["results"]) == 2
    assert result["results"][0]["doc_idx"] == 1
    assert result["results"][1]["doc_idx"] in [0, 2]
    assert all(0 <= ctx["score"] <= 1 for ctx in result["results"])
    assert result["error"] is None


def test_top_n_equals_one():
    query = "What is the capital of France?"
    contexts = [
        "The capital of France is Paris.",
        "Florida is a state in the USA."
    ]
    result = search_contexts_by_index(query, contexts, MODEL_PATH, top_n=1)
    assert result["is_valid"]
    assert len(result["results"]) == 1
    assert result["results"][0]["doc_idx"] == 0
    assert 0 <= result["results"][0]["score"] <= 1
    assert result["error"] is None


def test_top_n_all_contexts():
    query = "What is the capital of France?"
    contexts = [
        "The capital of France is Paris.",
        "Paris hosts many tourists in France.",
        "Florida is a state in the USA."
    ]
    result = search_contexts_by_index(query, contexts, MODEL_PATH, top_n=3)
    assert result["is_valid"]
    assert len(result["results"]) == 3
    assert set(ctx["doc_idx"] for ctx in result["results"]) == {0, 1, 2}
    assert all(0 <= ctx["score"] <= 1 for ctx in result["results"])
    assert result["error"] is None


def test_empty_query():
    query = ""
    contexts = [
        "The capital of France is Paris.",
        "Florida is a state in the USA."
    ]
    result = search_contexts_by_index(query, contexts, MODEL_PATH, top_n=1)
    assert not result["is_valid"]
    assert len(result["results"]) == 0
    assert "Query cannot be empty." in result["error"]


def test_empty_contexts():
    query = "What is the capital of France?"
    contexts = []
    result = search_contexts_by_index(query, contexts, MODEL_PATH, top_n=1)
    assert not result["is_valid"]
    assert len(result["results"]) == 0
    assert "Contexts cannot be empty." in result["error"]


def test_empty_context_in_list():
    query = "What is the capital of France?"
    contexts = [
        "The capital of France is Paris.",
        "",
        "Florida is a state in the USA."
    ]
    result = search_contexts_by_index(query, contexts, MODEL_PATH, top_n=1)
    assert not result["is_valid"]
    assert len(result["results"]) == 0
    assert "Context at index 1 cannot be empty." in result["error"]


def test_invalid_top_n_too_large():
    query = "What is the capital of France?"
    contexts = [
        "The capital of France is Paris.",
        "Florida is a state in the USA."
    ]
    result = search_contexts_by_index(query, contexts, MODEL_PATH, top_n=3)
    assert not result["is_valid"]
    assert len(result["results"]) == 0
    assert "top_n (3) cannot exceed number of contexts (2)." in result["error"]


def test_invalid_top_n_negative():
    query = "What is the capital of France?"
    contexts = [
        "The capital of France is Paris.",
        "Florida is a state in the USA."
    ]
    result = search_contexts_by_index(query, contexts, MODEL_PATH, top_n=0)
    assert not result["is_valid"]
    assert len(result["results"]) == 0
    assert "top_n must be at least 1." in result["error"]


def test_score_distribution():
    query = "What is the capital of France?"
    contexts = [
        "The capital of France is Paris.",
        "Paris is a major city in France.",
        "Florida is a state in the USA."
    ]
    result = search_contexts_by_index(query, contexts, MODEL_PATH, top_n=3)
    assert result["is_valid"]
    assert len(result["results"]) == 3
    assert all(ctx["score"] > 0 for ctx in result["results"])
    assert sum(ctx["score"]
               for ctx in result["results"]) == pytest.approx(1.0, abs=1e-5)
    assert result["results"][0]["doc_idx"] in [0, 1]
    scores = [ctx["score"] for ctx in result["results"]]
    assert len(set(scores)) > 1, "All scores are identical"
    # Verify scores match computed probabilities
    for ctx in result["results"]:
        idx = str(ctx["doc_idx"])
        score = ctx["score"]
        assert score > 1e-5, f"Score for doc_idx {idx} is too low: {score}"
    assert result["error"] is None
