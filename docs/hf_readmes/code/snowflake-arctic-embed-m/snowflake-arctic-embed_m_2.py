import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Snowflake/snowflake-arctic-embed-m')
model = AutoModel.from_pretrained(
    'Snowflake/snowflake-arctic-embed-m', add_pooling_layer=False)
model.eval()

query_prefix = 'Represent this sentence for searching relevant passages: '
queries = ['what is snowflake?', 'Where can I get the best tacos?']
queries_with_prefix = ["{}{}".format(query_prefix, i) for i in queries]
query_tokens = tokenizer(queries_with_prefix, padding=True,
                         truncation=True, return_tensors='pt', max_length=512)

documents = ['The Data Cloud!', 'Mexico City of Course!']
document_tokens = tokenizer(
    documents, padding=True, truncation=True, return_tensors='pt', max_length=512)

# Compute token embeddings
with torch.no_grad():
    query_embeddings = model(**query_tokens)[0][:, 0]
    document_embeddings = model(**document_tokens)[0][:, 0]


# normalize embeddings
query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
document_embeddings = torch.nn.functional.normalize(
    document_embeddings, p=2, dim=1)

scores = torch.mm(query_embeddings, document_embeddings.transpose(0, 1))
for query, query_scores in zip(queries, scores):
    doc_score_pairs = list(zip(documents, query_scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    # Output passages & scores
    print("Query:", query)
    for document, score in doc_score_pairs:
        print(score, document)
