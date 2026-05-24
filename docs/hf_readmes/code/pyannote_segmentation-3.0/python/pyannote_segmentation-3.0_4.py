from pyannote.audio.pipelines import OverlappedSpeechDetection
pipeline = OverlappedSpeechDetection(segmentation=model)
HYPER_PARAMETERS = {
  # remove overlapped speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-overlapped speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}
pipeline.instantiate(HYPER_PARAMETERS)
osd = pipeline("audio.wav")
# `osd` is a pyannote.core.Annotation instance containing overlapped speech regions