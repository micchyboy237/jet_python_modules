import logging
from jet.adapters.bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
try:
    from plotly.graph_objects import Figure
except ImportError:
    Figure = None
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def example_load_save_model():
    """Demonstrate saving and loading a BERTopic model."""
    logging.info("Creating and saving BERTopic model...")
    model = BERTopic(language="Dutch", embedding_model=None)
    model.save("test_model", serialization="pickle")
    logging.info("Loading saved model...")
    loaded_model = BERTopic.load("test_model")
    
    logging.info("Verifying model properties...")
    assert type(model) is type(loaded_model)
    assert model.language == loaded_model.language
    assert model.embedding_model == loaded_model.embedding_model
    assert model.top_n_words == loaded_model.top_n_words
    logging.info("Model save and load verified successfully")
    
    return loaded_model

def example_get_params():
    """Demonstrate retrieving BERTopic model parameters."""
    logging.info("Creating BERTopic model and retrieving parameters...")
    model = BERTopic()
    params = model.get_params()
    
    logging.info(f"Model parameters: {params}")
    assert not params["embedding_model"]
    assert not params["low_memory"]
    assert not params["nr_topics"]
    assert params["n_gram_range"] == (1, 1)
    assert params["min_topic_size"] == 10
    assert params["language"] == "english"
    logging.info("Parameter retrieval verified successfully")
    
    return params

def example_no_plotly():
    """Demonstrate handling visualization without Plotly."""
    logging.info("Creating BERTopic model for visualization test...")
    model = BERTopic(
        language="Dutch",
        embedding_model=None,
        min_topic_size=2,
        top_n_words=1,
        umap_model=BaseDimensionalityReduction(),
    )
    documents = ["hello", "hi", "goodbye", "goodbye", "whats up"] * 10
    logging.info("Fitting model with sample documents...")
    model.fit(documents)
    
    logging.info("Attempting to visualize topics...")
    try:
        fig = model.visualize_topics()
        fig.write_image(f"{OUTPUT_DIR}/fig.png")
        logging.info(f"Visualization image saved at: {OUTPUT_DIR}/fig.png")
        if Figure:
            assert isinstance(fig, Figure)
            logging.info("Visualization is a Plotly Figure")
        else:
            logging.info("Plotly not available, visualization skipped")
    except ImportError as e:
        logging.info(f"Expected ImportError caught: {str(e)}")
        assert "Plotly is required to use" in str(e)
    
    return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_load_save_model()
    example_get_params()
    example_no_plotly()
    logging.info("All other usage examples completed successfully.")
