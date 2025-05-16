import { pipeline, cos_sim } from '@xenova/transformers';

// Create a feature extraction pipeline
const extractor = await pipeline('feature-extraction', 'mixedbread-ai/mxbai-embed-large-v1', {
    quantized: false, // Comment out this line to use the quantized version
});

// Generate sentence embeddings
const docs = [
    'Represent this sentence for searching relevant passages: A man is eating a piece of bread',
    'A man is eating food.',
    'A man is eating pasta.',
    'The girl is carrying a baby.',
    'A man is riding a horse.',
]
const output = await extractor(docs, { pooling: 'cls' });

// Compute similarity scores
const [source_embeddings, ...document_embeddings ] = output.tolist();
const similarities = document_embeddings.map(x => cos_sim(source_embeddings, x));
console.log(similarities); // [0.7919578577247139, 0.6369278664248345, 0.16512018371357193, 0.3620778366720027]