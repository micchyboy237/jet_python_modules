from pyannote.audio.pipelines.utils.hook import ProgressHook
with ProgressHook() as hook:
    diarization = pipeline("audio.wav", hook=hook)