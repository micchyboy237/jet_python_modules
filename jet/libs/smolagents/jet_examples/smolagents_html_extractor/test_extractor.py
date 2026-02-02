import pytest
from extractor.tools import chunk_html, extract_relevant_content, format_final_results
from extractor.checkpoint import CheckpointManager
from pathlib import Path
import shutil

@pytest.fixture
def sample_html():
    return """
    <html><body>
    <h1>Test Document</h1>
    <p>This is a test paragraph about Python programming.</p>
    <p>Python is great for data science and web development.</p>
    <p>Artificial intelligence is transforming the world.</p>
    <p>Another sentence without any keywords.</p>
    </body></html>
    """.strip()


def test_chunk_html(sample_html):
    chunks = chunk_html(sample_html, window_size=100, overlap=30)
    assert len(chunks) >= 2
    assert all("text" in c for c in chunks)
    assert chunks[0]["start_char"] == 0
    # Use BeautifulSoup to get text length
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(sample_html, 'html.parser')
    plain_text = soup.get_text(separator=' ', strip=True)
    assert chunks[-1]["end_char"] <= len(plain_text)


def test_extract_relevant(sample_html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(sample_html, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    
    results = extract_relevant_content(text, query="python programming")
    assert len(results) > 0
    assert any("Python" in r["text"] for r in results)
    assert all(r["score"] >= 1 for r in results)


def test_format_results():
    fake_results = [
        {"text": "Python is great.", "score": 3, "relevance": "high"},
        {"text": "Another sentence.", "score": 1, "relevance": "medium"}
    ]
    output = format_final_results(fake_results)
    assert "Found relevant content" in output
    assert "PYTHON IS GREAT" in output.upper()


def test_checkpoint(tmp_path):
    checkpoint = CheckpointManager(str(tmp_path))
    
    results = [{"text": "test", "score": 1}]
    checkpoint.save_partial_results(results)
    checkpoint.save_progress(5, 20)
    
    loaded_results = checkpoint.load_results()
    loaded_progress = checkpoint.load_progress()
    
    assert len(loaded_results) == 1
    assert loaded_progress["processed_chunks"] == 5
    assert loaded_progress["total_chunks"] == 20


def test_full_pipeline(sample_html):
    from extractor.extractor import run_html_extraction_pipeline
    result = run_html_extraction_pipeline(
        sample_html,
        query="python",
        window_size=150,
        overlap=40,
        resume=False
    )
    assert "Python" in result
