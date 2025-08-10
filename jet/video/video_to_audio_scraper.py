import os
from pydub import AudioSegment

from jet.data.utils import generate_hash
from jet.video.youtube.youtube_scrape_info import transcribe_in_segments


def convert_video_to_audio(video_path):
    print(f"Optimizing video file {video_path}")
    audio = AudioSegment.from_file(video_path)

    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Normalize audio
    normalized_audio = audio.apply_gain(-audio.max_dBFS)

    # Determine the output path
    file_root, file_ext = os.path.splitext(video_path)
    if file_ext.lower() in ['.mp4', '.mov']:
        converted_audio_path = file_root + '.mp3'
    else:
        raise ValueError("Unsupported file format")

    normalized_audio.export(converted_audio_path, format='mp3')

    return converted_audio_path


def transcribe_audio(audio_path):
    transcription_stream = transcribe_in_segments(audio_path)
    transcriptions = []

    for current_segments in transcription_stream:
        for segment in current_segments:
            print(f"Segment:\n{segment}")
            if not segment['text']:
                continue

            transcription = {
                "id": generate_hash({
                    "start": segment['start'],
                    "end": segment['end'],
                }),
                "seek": segment['seek'],
                "start": segment['start'],
                "end": segment['end'],
                "text": segment['text'],
                "eval": {
                    "avg_logprob": segment['avg_logprob'],
                    "compression_ratio": segment['compression_ratio'],
                    "no_speech_prob": segment['no_speech_prob'],
                },
                "words": segment['words'],
            }

            transcriptions.append(transcription)

            yield transcription
            # batch_items.append(transcription)

            # if len(batch_items) >= batch_size:
            #     save_data(transcription_file_path, batch_items)
            #     batch_items = []

    # if batch_items:
    #     save_data(transcription_file_path, batch_items)


if __name__ == '__main__':
    video_path = '/Users/jethroestrada/Desktop/External_Projects/GPT/xturing-jet-examples/data/scrapers/test/initial-interview-1.mov'
    converted_audio_path = convert_video_to_audio(video_path)
    print(
        f"Converted video file {video_path} to audio file {converted_audio_path}")
