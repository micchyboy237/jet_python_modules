from pyannote.audio import Inference
from pyannote.core import Segment
inference = Inference(model, window="whole")
excerpt = Segment(13.37, 19.81)
embedding = inference.crop("audio.wav", excerpt)
# `embedding` is (1 x D) numpy array extracted from the file excerpt.