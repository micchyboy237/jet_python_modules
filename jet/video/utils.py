import json
from typing import Optional
import urllib.request
import yt_dlp
import os
from pydub import AudioSegment
from pydub.utils import make_chunks
from jet.data.utils import generate_hash
from jet.models.model_registry.transformers.speech_to_text.whisper_model_registry import WhisperModelRegistry
from jet.wordnet.sentence import split_sentences
import numpy as np
import noisereduce as nr
from pydub.effects import normalize
from tqdm import tqdm


def download(url: str, target_path: str):
    with urllib.request.urlopen(url) as source, open(target_path, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))


# def download_audio(video_url: str, audio_dir: str, audio_format: str) -> str:
#     ydl_opts = {
#         'format': 'bestaudio/best',
#         'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': audio_format,
#             'preferredquality': '320',
#         }],
#         'outtmpl': audio_dir + '/audio.%(ext)s',
#     }

#     print(f"Downloading audio from {video_url}")

#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             info_dict = ydl.extract_info(video_url, download=True)
#             audio_file = ydl.prepare_filename(info_dict).replace(
#                 'webm', audio_format).replace('m4a', audio_format)
#             return audio_file
#     except Exception as e:
#         print(f"Failed to download audio: {e}")
#         return None

def download_audio(video_url: str, audio_dir: str, audio_format: str) -> Optional[str]:
    """
    Download audio from a video URL and save it to the specified directory.

    Args:
        video_url: URL of the video to download audio from.
        audio_dir: Directory to save the audio file.
        audio_format: Desired audio format (e.g., 'mp3', 'aac').

    Returns:
        Path to the downloaded audio file, or None if the download fails.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
            'preferredquality': '320',
        }],
        'paths': {'home': audio_dir},
        'outtmpl': {'default': os.path.join(audio_dir, 'audio.%(ext)s')},
    }

    print(f"Downloading audio from video: {video_url}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            audio_file = ydl.prepare_filename(info_dict)
            # Ensure the extension matches the requested audio format
            audio_file = os.path.splitext(audio_file)[0] + f'.{audio_format}'
            if not os.path.exists(audio_file):
                raise FileNotFoundError(
                    f"Audio file not found at {audio_file}")
            print(f"Downloaded audio to {audio_file}")
            return audio_file
    except (yt_dlp.utils.DownloadError, FileNotFoundError) as e:
        print(f"Failed to download audio: {e}")
        return None


# Function to apply noise reduction and normalization
def process_audio(audio_path):
    print(f"Optimizing audio file {audio_path}")
    audio = AudioSegment.from_file(audio_path)

    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Convert to NumPy array
    audio_np = np.array(audio.get_array_of_samples())

    # Use the first 5000 samples for noise profile
    noise_profile = audio_np[:5000]

    # Apply noise reduction
    reduced_noise_audio_np = nr.reduce_noise(
        y=audio_np,
        sr=audio.frame_rate,
        y_noise=noise_profile  # Correct parameter name here
    )

    # Convert back to AudioSegment
    reduced_noise_audio = AudioSegment(
        reduced_noise_audio_np.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=1  # Set channels to 1 as it's now mono
    )

    # Normalize audio
    normalized_audio = normalize(reduced_noise_audio)

    processed_audio_path = audio_path.replace('.mp4', '_processed.mp3')
    normalized_audio.export(processed_audio_path, format='mp3')

    return processed_audio_path


def split_audio_and_transcribe(audio_path, chunk_length_s, overlap_s, audio_codec_ext, chunk_dir, transcription_dir, model_size):
    print(
        f"Splitting audio file {audio_path} into chunks of {chunk_length_s} ms")
    print(f"Overlap between chunks: {overlap_s} ms")

    if not os.path.exists(chunk_dir):
        os.makedirs(chunk_dir)
    if not os.path.exists(transcription_dir):
        os.makedirs(transcription_dir)

    # Get last chunk id and last transcription id then compute next chunk id
    last_chunk_id = -1
    for file in os.listdir(transcription_dir):
        if file.endswith(".json"):
            transcription_id = int(file.split('.')[0].replace(
                'chunk', '').replace('_transcription', ''))
            if transcription_id > last_chunk_id:
                last_chunk_id = transcription_id

    print(f"Last chunk id: {last_chunk_id}")

    # Load the audio file
    audio = AudioSegment.from_file(audio_path)

    # Make chunks of chunk_length_s milliseconds
    chunk_length_ms = chunk_length_s * 1000
    chunks = make_chunks(audio, chunk_length_ms)
    print(f"Total number of chunks: {len(chunks)}")

    # Initialize the elapsed time before the first chunk
    offset_seconds = 0

    # Adjust the chunking to create an overlap
    for i in range(len(chunks)):
        if i >= last_chunk_id + 1:
            print(f"Processing chunk {i}")
            chunk = chunks[i]

            # Export each chunk as a separate file and transcribe them
            chunk_name = f"{chunk_dir}/chunk{i}.{audio_codec_ext}"
            print(f"Exporting {chunk_name}")
            chunk.export(chunk_name, format=audio_codec_ext)

            # Delete previous chunk
            chunk_to_remove = f"{chunk_dir}/chunk{i-1}.{audio_codec_ext}"
            if i > 0 and os.path.exists(chunk_to_remove):
                os.remove(chunk_to_remove)

            # Transcribe the chunk immediately after exporting
            result = transcribe_audio_chunk(chunk_name, str(
                i), offset_seconds, model_size)

            offset_seconds = result['last_segment_end']

        # Update the elapsed time for the next chunk
        if i > 0:
            offset_seconds -= overlap_s

    print(
        f"Audio split into chunks of {chunk_length_s} ms each and transcribed.")


def transcribe_audio_chunk(chunk_path, chunk_id, offset_seconds=0, model_size="small"):
    print(f"Transcribing audio file {chunk_path} using Whisper-{model_size}")

    registry = WhisperModelRegistry()
    model = registry.load_model(model_size)
    result = model.transcribe(chunk_path)

    transcription_list = []

    for segment in result["segments"]:
        current_text = segment['text'].strip()
        avg_logprob = f"{segment['avg_logprob']:.2f}"
        compression_ratio = f"{segment['compression_ratio']:.2f}"
        no_speech_prob = f"{segment['no_speech_prob']:.2f}"

        # Skip if current segment text matches the latest in transcription_list
        if transcription_list and transcription_list[-1]["text"] == current_text:
            continue

        # Check if combining current text with the last text matches the previous two segments
        if len(transcription_list) >= 2 and \
           transcription_list[-1]["text"] + current_text == transcription_list[-2]["text"] + transcription_list[-1]["text"]:
            transcription_list.pop()  # Remove the last text
            continue

        start_time_in_seconds = int(segment['start'] + offset_seconds)
        end_time_in_seconds = int(segment['end'] + offset_seconds)

        # Convert seconds to minutes and seconds
        start_min, start_sec = divmod(start_time_in_seconds, 60)
        end_min, end_sec = divmod(end_time_in_seconds, 60)

        # Format the start and end times
        start_time_formatted = f"{start_min:02d}:{start_sec:02d}"
        end_time_formatted = f"{end_min:02d}:{end_sec:02d}"

        # Calculate and format the duration
        duration_sec = end_time_in_seconds - start_time_in_seconds
        duration_min, duration_sec = divmod(duration_sec, 60)
        duration_formatted = f"{duration_min:02d}:{duration_sec:02d}"

        segment_data = {
            "start_min": start_time_formatted,
            "end_min": end_time_formatted,
            "text": current_text,
            "avg_logprob": avg_logprob,
            "compression_ratio": compression_ratio,
            "no_speech_prob": no_speech_prob,
            "duration": duration_formatted,
        }

        # Include confidence and speaker data if available
        if 'confidence' in segment:
            segment_data['confidence'] = segment['confidence']
        if 'speaker' in segment:
            segment_data['speaker'] = segment['speaker']

        transcription_list.append(segment_data)
        previous_segment_data = segment_data

    # get chunk_path parent.parent dir
    audio_dir = os.path.dirname(os.path.dirname(chunk_path))
    output_dir = f'{audio_dir}/transcriptions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Deduplicate the transcription list based on the text, removing the earlier segments with the same text
    print(
        f"Transcription list before deduplication: {len(transcription_list)}")
    transcription_list = deduplicate_transcription_list(transcription_list)
    print(f"Transcribed {len(transcription_list)} segments")

    # Save the transcribed text as a JSON file
    json_filename = f"{output_dir}/chunk{chunk_id}_transcription.json"
    with open(json_filename, "w", encoding="utf-8") as json_file:
        json.dump(transcription_list, json_file, indent=4, ensure_ascii=False)

    print(f"Chunk {chunk_id} transcribed and saved to {json_filename}")

    return {
        "last_segment_start": start_time_in_seconds,
        "last_segment_end": end_time_in_seconds
    }


def has_video(video_id):
    video_dir = f'data/scrapers/audio/{video_id}'
    video_path = f'{video_dir}/video.mp4'
    return os.path.exists(video_path)


def has_audio(video_id):
    audio_dir = f'data/scrapers/audio/{video_id}'
    video_path = f'{audio_dir}/video_processed.mp3'
    return os.path.exists(video_path)


def has_transcriptions(video_id):
    audio_dir = f'data/scrapers/audio/{video_id}'
    transcription_dir = f'{audio_dir}/transcriptions/'

    # Check if there are existing *_transcription.json files
    for file in os.listdir(transcription_dir):
        if file.endswith("_transcription.json"):
            return True
    return False


def has_chunks(video_id):
    audio_dir = f'data/scrapers/audio/{video_id}'
    chunk_dir = f'{audio_dir}/chunks/'

    # Check if there are existing chunk*.mp3 files
    for file in os.listdir(chunk_dir):
        if file.endswith(".mp3"):
            return True
    return False


def is_done(video_id):
    return has_transcriptions(video_id) and not has_chunks(video_id)


def main(video_id, chunk_length_s=300, overlap_s=10, video_codec_ext='mp4', audio_codec_ext='mp3', model_size="small"):
    audio_dir = f'data/scrapers/audio/{video_id}'
    chunk_dir = f'{audio_dir}/chunks'
    transcription_dir = f'{audio_dir}/transcriptions'
    video_path = f'{audio_dir}/video.mp4'
    # Create chunk_dir and transcription_dir if they don't exist
    if not os.path.exists(chunk_dir):
        os.makedirs(chunk_dir)
    if not os.path.exists(transcription_dir):
        os.makedirs(transcription_dir)

    if is_done(video_id):
        print("Already done!")
    else:
        if not has_audio(video_id):
            video_url = f'https://www.youtube.com/watch?v={video_id}'
            if not has_video(video_id):
                audio_path = download_audio(
                    video_url, audio_dir, video_codec_ext)
            else:
                audio_path = video_path
            audio_path = process_audio(audio_path)
        else:
            audio_path = video_path

        # Split the audio file into chunks
        split_audio_and_transcribe(
            audio_path, chunk_length_s, overlap_s, audio_codec_ext, chunk_dir, transcription_dir, model_size)

        print("Done transcribing all!")


def deduplicate_transcription_list(transcription_list):
    seen_texts = set()
    deduplicated_list = []

    # Iterate backwards through the list to keep the latest copy of each text
    for segment in transcription_list:
        if segment["text"] not in seen_texts:
            deduplicated_list.append(segment)
            seen_texts.add(segment["text"])

    return deduplicated_list


def deduplicate_all_transcriptions(audio_files):
    for file in tqdm(audio_files, desc="Deduplicating transcriptions"):
        with open(file, 'r', encoding='utf-8') as f:
            transcriptions = json.load(f)

        # Loop through the transcriptions by getting current (i) and next (i+1) items, check if they are the same "chapter_title"
        i = 0
        while i < len(transcriptions) - 1:
            current_item = transcriptions[i]
            next_item = transcriptions[i + 1]
            has_same_transcription = current_item["text"] == next_item["text"]
            has_same_chapter_title = current_item["chapter_title"] == next_item["chapter_title"]
            if has_same_transcription and has_same_chapter_title:
                transcriptions.pop(i)
            else:
                i += 1

        with open(file, 'w', encoding='utf-8') as f:
            json.dump(transcriptions, f, indent=2, ensure_ascii=False)


def convert_time_to_minutes(time_str):
    # Split the time string into hours and minutes and convert to integers
    hours, minutes = map(int, time_str.split(":"))
    # Convert hours to minutes and add to minutes
    return hours * 60 + minutes


def combine_all(output_path):
    audio_base_dir = 'data/scrapers/audio'
    # deduplicate_all_transcriptions(audio_base_dir)

    # Get all the video ids
    video_ids = [name for name in os.listdir(
        audio_base_dir) if os.path.isdir(os.path.join(audio_base_dir, name))]
    transcriptions = []

    # Iterate over each video ID's transcription directory
    for video_id in video_ids:
        audio_dir = os.path.join(audio_base_dir, video_id)
        transcription_dir = os.path.join(audio_dir, 'transcriptions')

        # Check if the directory exists
        if os.path.exists(transcription_dir):
            # Check if there are existing *_transcription.json files
            for file in sorted(os.listdir(transcription_dir)):
                if file.endswith("_transcription.json"):
                    with open(os.path.join(transcription_dir, file), 'r', encoding='utf-8') as f:
                        transcription_list = json.load(f)

                        # Sort the transcriptions by converting start_min to total minutes
                        transcription_list.sort(
                            key=lambda x: convert_time_to_minutes(x['start_min']))

                        combined_texts = []
                        # Add video ID to each transcription
                        for transcription in transcription_list:
                            combined_texts.append(transcription['text'])

                        combined_text = ' '.join(combined_texts)
                        splitted_sentences = split_sentences(combined_text)
                        for sentence in splitted_sentences:
                            transcription = {
                                'id': generate_hash(sentence),
                                'text': sentence,
                                'category': video_id
                            }
                            transcriptions.append(transcription)

    # Save the transcribed text as a JSON file
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(transcriptions, json_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # 5 minutes chunks
    model_size = "medium"
    chunk_length_s = 300
    overlap_s = 10
    audio_codec_ext = 'mp3'

    video_ids = [
        'c2iZXYX_3UI',
        'HSuFxLXnXDQ',
        'Q4yRe7Bk1Uo',
        '8UEHsTNLNhY',
        'qWKMMvyizNI',
        'ICN5TUg_P3U',
        'UWeKxl_mI_8'
    ]

    for video_id in video_ids:
        main(video_id, model_size=model_size)

    combine_all(
        'server/static/models/dost-asti-gpt2/base_model/datasets/conversations.json')


# if __name__ == '__main__':
#     model_size = "small"
#     chunk_length_s = 300
#     overlap_s = 10
#     audio_codec_ext = 'mp3'

#     video_id = 'c2iZXYX_3UI'
#     audio_dir = f'data/scrapers/audio/{video_id}'
#     chunk_dir = f'{audio_dir}/chunks'
#     transcription_dir = f'{audio_dir}/transcriptions'
#     video_path = f'{audio_dir}/video_processed.mp3'

#     split_audio_and_transcribe(
#         f'data/scrapers/audio/{video_id}/video_processed.mp3', chunk_length_s, overlap_s, audio_codec_ext, chunk_dir, transcription_dir, model_size)

# if __name__ == '__main__':
#     deduplicate_all_transcriptions('data/scrapers/audio/c2iZXYX_3UI')
#     combine_all('server/static/models/dost-asti-gpt2/base_model/datasets/conversations.json')
#     print("Done!")
