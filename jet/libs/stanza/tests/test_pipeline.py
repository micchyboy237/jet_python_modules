import pytest
from jet.libs.stanza.pipeline import StanzaPipelineCache

@pytest.fixture
def pipeline_cache():
    # Given: A fresh StanzaPipelineCache instance
    cache = StanzaPipelineCache()
    cache.clear_cache()  # Ensure clean state before each test
    yield cache
    # Cleanup: Clear cache after each test
    cache.clear_cache()

class TestStanzaPipelineCache:
    def test_singleton_instance(self, pipeline_cache):
        # Given: A StanzaPipelineCache instance
        first_instance = pipeline_cache
        # When: Creating another instance
        second_instance = StanzaPipelineCache()
        # Then: Both instances should be the same
        result = first_instance is second_instance
        expected = True
        assert result == expected, "StanzaPipelineCache should return the same instance for multiple calls"

    def test_get_pipeline_creates_single_pipeline(self, pipeline_cache):
        # Given: An empty cache
        # When: Requesting a pipeline with specific configuration
        pipeline = pipeline_cache.get_pipeline(lang="en", processors="tokenize,pos", use_gpu=False, verbose=True)
        # Then: The pipeline should be stored and match the requested configuration
        result = pipeline_cache._pipeline
        expected = pipeline
        assert result == expected, "Cache should store the created pipeline"
        assert pipeline.lang == "en", "Pipeline language should be 'en'"
        assert set(pipeline.processors.keys()) == {"tokenize", "pos", "mwt"}, "Pipeline processors should match requested"

    def test_get_pipeline_replaces_pipeline(self, pipeline_cache):
        # Given: A cache with an existing pipeline
        first_pipeline = pipeline_cache.get_pipeline(lang="en", processors="tokenize", use_gpu=False, verbose=True)
        # When: Requesting a pipeline with different configuration
        second_pipeline = pipeline_cache.get_pipeline(lang="fr", processors="tokenize,pos", use_gpu=True, verbose=True)
        # Then: The cache should store only the new pipeline
        result = pipeline_cache._pipeline
        expected = second_pipeline
        assert result == expected, "Cache should replace old pipeline with new one"
        assert result != first_pipeline, "New pipeline should differ from old one"
        assert pipeline_cache._config == ("fr", "tokenize,pos", True), "Config should match new pipeline"

    def test_clear_cache(self, pipeline_cache):
        # Given: A cache with a pipeline
        pipeline_cache.get_pipeline(lang="en", processors="tokenize", use_gpu=False, verbose=True)
        # When: Clearing the cache
        pipeline_cache.clear_cache()
        # Then: The pipeline and config should be None
        result_pipeline = pipeline_cache._pipeline
        result_config = pipeline_cache._config
        expected = None
        assert result_pipeline == expected, "Pipeline should be None after clear_cache"
        assert result_config == expected, "Config should be None after clear_cache"

    def test_get_pipeline_reuses_same_config(self, pipeline_cache):
        # Given: A cache with a pipeline
        first_pipeline = pipeline_cache.get_pipeline(lang="en", processors="tokenize,pos", use_gpu=False, verbose=True)
        # When: Requesting a pipeline with the same configuration
        second_pipeline = pipeline_cache.get_pipeline(lang="en", processors="tokenize,pos", use_gpu=False, verbose=True)
        # Then: The same pipeline instance should be returned
        result = second_pipeline
        expected = first_pipeline
        assert result == expected, "Cache should reuse the same pipeline for identical configuration"
