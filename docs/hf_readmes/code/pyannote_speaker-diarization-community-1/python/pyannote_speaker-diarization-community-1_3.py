waveform, sample_rate = torchaudio.load("audio.wav")
output = pipeline({"waveform": waveform, "sample_rate": sample_rate})