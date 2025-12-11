# load pipeline from disk (works without internet connection)
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained('/path/to/directory/pyannote-speaker-diarization-community-1')

# run the pipeline locally on your computer
output = pipeline("audio.wav")