import pytest
from jet.audio.speech.wav_utils import _validate_audio_data, save_wav_file, get_wav_bytes, get_wav_fileobj

# Assuming these constants are defined in your module
CHANNELS = 1
SAMPLE_RATE = 16000
DTYPE = 'int16'
