import json
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

def extract_speech_timestamps(audio_file):
    model = load_silero_vad()
    wav = read_audio(audio_file, sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=0.5,  # Speech if prob > 0.5
        min_silence_duration_ms=500,  # 0.5s silence to end segment
        max_speech_duration_s=10,  # Split long speech
        return_seconds=True,  # Output: [{'start': 1.2, 'end': 3.4}, ...]
        window_size_samples=512  # 32ms @16kHz (default)
    )
    return speech_timestamps

if __name__ == "__main__":
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_212124.wav"
    speech_timestamps = extract_speech_timestamps(audio_file)
    print(json.dumps(speech_timestamps, indent=2))  # [{'start': 0.0, 'end': 2.5}, {'start': 5.0, 'end': 7.8}]