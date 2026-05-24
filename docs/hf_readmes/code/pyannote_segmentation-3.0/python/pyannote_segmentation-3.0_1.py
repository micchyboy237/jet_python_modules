# waveform (first row)
duration, sample_rate, num_channels = 10, 16000, 1
waveform = torch.randn(batch_size, num_channels, duration * sample_rate) 

# powerset multi-class encoding (second row)
powerset_encoding = model(waveform)

# multi-label encoding (third row)
from pyannote.audio.utils.powerset import Powerset
max_speakers_per_chunk, max_speakers_per_frame = 3, 2
to_multilabel = Powerset(
    max_speakers_per_chunk, 
    max_speakers_per_frame).to_multilabel
multilabel_encoding = to_multilabel(powerset_encoding)