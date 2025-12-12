from pyannote.audio import Inference
inference = Inference(model, window="sliding",
                      duration=3.0, step=1.0)
embeddings = inference("audio.wav")
# `embeddings` is a (N x D) pyannote.core.SlidingWindowFeature
# `embeddings[i]` is the embedding of the ith position of the 
# sliding window, i.e. from [i * step, i * step + duration].