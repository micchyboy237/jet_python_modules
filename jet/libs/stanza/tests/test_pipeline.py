from typing import List, Optional
import pytest
import stanza
from jet.libs.stanza.pipeline import StanzaPipelineCache
from rag_stanza import build_stanza_pipeline

@pytest.fixture
def clear_cache():
    """Fixture to clear the pipeline cache before and after each test."""
    cache = StanzaPipelineCache()
    cache.clear_cache()
    yield
    cache.clear_cache()

class TestStanzaPipelineCache:
    def test_pipeline_caching_same_config(self, clear_cache):
        """Given the same configuration, When requesting the pipeline twice, Then the same instance is returned."""
        # Given
        expected: Optional[stanza.Pipeline] = None
        
        # When
        pipeline1 = build_stanza_pipeline()
        pipeline2 = build_stanza_pipeline()
        result = pipeline1 is pipeline2  # Check for object identity
        
        # Then
        expected = True
        assert result == expected, "Pipelines with the same config should be identical (cached)."

    def test_pipeline_caching_different_configs(self, clear_cache):
        """Given different configurations, When requesting pipelines, Then different instances are returned."""
        # Given
        cache = StanzaPipelineCache()
        expected: Optional[stanza.Pipeline] = None
        
        # When
        pipeline1 = cache.get_pipeline(lang="en", processors="tokenize,pos")
        pipeline2 = cache.get_pipeline(lang="en", processors="tokenize,ner")
        result = pipeline1 is pipeline2
        
        # Then
        expected = False
        assert result == expected, "Pipelines with different configs should not be identical."

    def test_pipeline_caching_thread_safety(self, clear_cache):
        """Given concurrent pipeline requests, When fetching pipelines, Then the same instance is returned."""
        # Given
        import threading
        results: List[bool] = []
        pipeline_ref: Optional[stanza.Pipeline] = None
        expected = True
        
        def fetch_pipeline():
            pipeline = build_stanza_pipeline()
            results.append(pipeline is pipeline_ref)
        
        # When
        pipeline_ref = build_stanza_pipeline()
        threads = [threading.Thread(target=fetch_pipeline) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        result = all(results)
        
        # Then
        assert result == expected, "Concurrent pipeline requests should return the same cached instance."
