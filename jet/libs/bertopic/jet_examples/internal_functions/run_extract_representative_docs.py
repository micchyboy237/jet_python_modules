def example__extract_representative_docs() -> Union[List[str], List[List[int]]]:
    """Demonstrates the _extract_representative_docs method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    topics = model.get_topics()
    result = model._extract_representative_docs(
        c_tf_idf=model.c_tf_idf_,
        documents=df,
        topics=topics,
        nr_samples=500,
        nr_repr_docs=3,
        diversity=0.5
    )
    logger.success(f"_extract_representative_docs: {result}")
    return result