# instantiate the pipeline
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")

# run the pipeline on an audio file
diarization = pipeline("audio.wav")

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)