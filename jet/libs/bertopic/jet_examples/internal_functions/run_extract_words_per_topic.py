def example__extract_words_per_topic() -> Mapping[str, List[Tuple[str, float]]]:
    """Demonstrates the _extract_words_per_topic method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS, "Topic": model.topics_})
    words = model.vectorizer_model.get_feature_names_out()
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    result = model._extract_words_per_topic(
        words=words,
        documents=df,
        c_tf_idf=model.c_tf_idf_,
        fine_tune_representation=True,
        calculate_aspects=True,
        embeddings=embeddings
    )
    logger.success(f"_extract_words_per_topic: {result}")
    return result