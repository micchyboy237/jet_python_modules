from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/stsb-roberta-base')
scores = model.predict([('Sentence 1', 'Sentence 2'), ('Sentence 3', 'Sentence 4')])