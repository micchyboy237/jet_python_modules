from pyannote.audio.pipelines.utils.hook import ProgressHook
with ProgressHook() as hook:
    output = pipeline("audio.wav", hook=hook)