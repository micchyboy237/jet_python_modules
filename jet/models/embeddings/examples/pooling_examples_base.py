from sentence_transformers import SentenceTransformer, models

word_embedding_model = models.Transformer('bert-base-uncased')
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
