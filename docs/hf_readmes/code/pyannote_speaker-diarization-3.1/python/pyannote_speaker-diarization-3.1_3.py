waveform, sample_rate = torchaudio.load("audio.wav")
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})