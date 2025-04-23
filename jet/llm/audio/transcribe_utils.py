import os
import asyncio
from typing import Optional
from jet.file.utils import save_file
from jet.logger import logger
from pydub import AudioSegment
from faster_whisper import WhisperModel
from tqdm import tqdm

last_transcribed_end_time = 0.0


def is_audio_file(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith((".mp3", ".wav"))


def is_audio_dir(path: str) -> bool:
    return os.path.isdir(path)


def is_audio_file_or_dir(path: str) -> bool:
    return is_audio_file(path) or is_audio_dir(path)


def transcribe_chunk(model, audio_chunk, time_offset=0):
    global last_transcribed_end_time
    segments, info = model.transcribe(audio_chunk, beam_size=5)

    for segment in segments:
        start_time = segment.start + time_offset
        end_time = segment.end + time_offset
        if start_time < last_transcribed_end_time:
            continue

        item = {
            "start": start_time,
            "end": end_time,
            "text": segment.text.strip(),
            "tokens": segment.tokens,
            "avg_logprob": segment.avg_logprob,
            "no_speech_prob": segment.no_speech_prob,
            "compression_ratio": segment.compression_ratio
        }

        print(f"[{item['start']}s -> {item['end']}s] {item['text']}")
        yield item

        last_transcribed_end_time = end_time


def transcribe_in_buffers(audio_file, output_dir, chunk_duration_ms=30000, overlap_ms=1000):
    global last_transcribed_end_time
    last_transcribed_end_time = 0.0

    audio = AudioSegment.from_file(audio_file)
    original_extension = os.path.splitext(audio_file)[1].lower().lstrip(".")
    export_format = "mp3" if original_extension == "mp3" else "wav"

    model = WhisperModel("small", compute_type="int8")
    total_duration = len(audio)
    print(f"Total audio duration: {total_duration / 1000} seconds")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_file = os.path.join(output_dir, f"{audio_base_name}.json")

    if os.path.exists(output_file):
        os.remove(output_file)

    step_size = chunk_duration_ms - overlap_ms
    num_chunks = (total_duration + step_size - 1) // step_size

    results = []
    with tqdm(total=num_chunks, desc="Transcribing", unit="chunk") as pbar:
        for start_ms in range(0, total_duration, step_size):
            end_ms = min(start_ms + chunk_duration_ms, total_duration)
            audio_chunk = audio[start_ms:end_ms]

            chunk_file = os.path.join(
                output_dir, f"temp_chunk.{original_extension}")
            audio_chunk.export(chunk_file, format=export_format)

            time_offset = start_ms / 1000.0
            for item in transcribe_chunk(model, chunk_file, time_offset=time_offset):
                results.append(item)
                save_file(results, output_file, verbose=False)

            os.remove(chunk_file)
            pbar.update(1)

    logger.success(f"Transcriptions ({len(results)}) saved to {output_file}")


def transcribe_file(audio_file: str, output_dir: str, *, chunk_duration_ms: int = 30000, overlap_ms: int = 1000, remove_audio: bool = False):
    print(f"Transcribing audio file: {audio_file}")
    print("Transcribing audio in buffers...")
    transcribe_in_buffers(audio_file, output_dir,
                          chunk_duration_ms, overlap_ms)
    if remove_audio:
        os.remove(audio_file)
        print(f"Deleted {audio_file} after transcription.")
    else:
        print(
            f"Transcription complete. Original audio file '{audio_file}' was NOT removed.")


async def transcribe_file_async(audio_file: str, output_dir: str, *, chunk_duration_ms: int = 30000, overlap_ms: int = 1000, remove_audio: bool = False):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, transcribe_file, audio_file, output_dir)
    logger.success(f"Completed transcription for {audio_file}")


def transcribe_files(path: str, output_dir: str, *, chunk_duration_ms: int = 30000, overlap_ms: int = 1000, remove_audio: bool = False):
    audio_files = []
    if is_audio_dir(path):
        print(f"'{path}' is a directory. Scanning recursively for audio files...")
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith((".mp3", ".wav")):
                    audio_files.append(os.path.join(root, f))
    elif is_audio_file(path):
        audio_files.append(path)
    else:
        print(f"Path '{path}' is not a valid audio file or directory.")
        return
    print(f"Found {len(audio_files)} audio file(s) to transcribe.")
    for audio_path in audio_files:
        relative_path = os.path.relpath(
            audio_path, path if is_audio_dir(path) else os.path.dirname(path))
        file_output_dir = os.path.join(
            output_dir, os.path.dirname(relative_path))
        transcribe_file(audio_path, file_output_dir,
                        chunk_duration_ms=chunk_duration_ms,
                        overlap_ms=overlap_ms,
                        remove_audio=remove_audio)


def combine_audio_files(file_paths: list[str], output_file: str) -> Optional[str]:
    if not file_paths:
        logger.warning("No audio files to combine")
        return None
    try:
        combined = AudioSegment.empty()
        for file_path in file_paths:
            if os.path.exists(file_path):
                audio = AudioSegment.from_mp3(file_path)
                combined += audio
            else:
                logger.warning(f"Audio file not found: {file_path}")
        if not combined:
            logger.warning("No valid audio segments to combine")
            return None
        combined.export(output_file, format="mp3", bitrate="64k")
        logger.success(f"Combined audio saved: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error combining audio files: {e}")
        return None


async def combine_audio_files_async(file_paths: list[str], output_file: str) -> Optional[str]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, combine_audio_files, file_paths, output_file)

if __name__ == "__main__":
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/agent/tts_Interviewer_20250423_231046_181704_Interviewer_Lets_begin_the_i.mp3"
    output_dir = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "generated", os.path.splitext(os.path.basename(__file__))[0])
    transcribe_files(audio_file, output_dir, chunk_duration_ms=30000,
                     overlap_ms=1000, remove_audio=False)
