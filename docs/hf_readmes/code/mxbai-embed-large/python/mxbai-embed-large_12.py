from mixedbread_ai.client import MixedbreadAI, EncodingFormat
from sklearn.metrics.pairwise import cosine_similarity
import os

mxbai = MixedbreadAI(api_key="{MIXEDBREAD_API_KEY}")

english_sentences = [
    'What is the capital of Australia?',
    'Canberra is the capital of Australia.'
] 

res = mxbai.embeddings(
     input=english_sentences,
     model="mixedbread-ai/mxbai-embed-large-v1",
     normalized=True,
     encoding_format=[EncodingFormat.FLOAT, EncodingFormat.UBINARY, EncodingFormat.INT_8],
     dimensions=512
)

encoded_embeddings = res.data[0].embedding
print(res.dimensions, encoded_embeddings.ubinary, encoded_embeddings.float_, encoded_embeddings.int_8)